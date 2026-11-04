//! Python module `mahad_tok`.
//!
//! The core `tokenizer` crate does not depend on PyO3. This cdylib holds the
//! only Python types. `Encoding` owns an [`Arc`] of a [`RankTokenizer`] so
//! rayon workers and other Python threads can share one vocab.
//!
//! PyO3 0.30.1 (docs.rs): arguments and the module are [`Bound`], not the
//! deprecated GIL refs (`&PyAny`). Releasing the interpreter is
//! [`Python::detach`] (the old name was `Python::allow_threads`). The closure
//! must be [`Ungil`], which on stable follows `Send`: no `Bound` and no new
//! Python object inside it. [`TokenizerError`] becomes a Python `ValueError`
//! only after `detach` returns, because `PyValueError::new_err` needs the
//! interpreter.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use tokenizer::{RankTokenizer, TokenizerError};

/// How `encode` treats registered special tokens.
///
/// `None` and an empty collection are tiktoken's default `allowed_special`
/// (an empty set): every registered special is disallowed and raises.
/// `"all"` is tiktoken's `allowed_special="all"`.
enum SpecialMode {
    All,
    Only(HashSet<String>),
}

/// One rayon pool per thread count. Building a pool starts OS threads, and
/// tiktoken keeps its pool across calls. The map is locked only to look the
/// pool up. Encode runs after the guard is dropped, and it does not run while
/// the GIL is held: the caller detaches first.
fn cached_pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools
        .lock()
        .map_err(|err| PyValueError::new_err(format!("thread pool lock: {err}")))?;
    if let Some(pool) = guard.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|err| PyValueError::new_err(format!("failed to build thread pool: {err}")))?;
    let pool = Arc::new(pool);
    guard.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

fn thread_count(requested: Option<usize>) -> PyResult<usize> {
    match requested {
        Some(0) => Err(PyValueError::new_err("num_threads must be at least 1")),
        Some(n) => Ok(n),
        None => Ok(std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)),
    }
}

fn special_mode(allowed: Option<&Bound<'_, PyAny>>) -> PyResult<SpecialMode> {
    let Some(obj) = allowed else {
        return Ok(SpecialMode::Only(HashSet::new()));
    };
    if obj.is_none() {
        return Ok(SpecialMode::Only(HashSet::new()));
    }
    // pyo3 0.30 does not implement `FromPyObject` for `&str` (that borrow
    // used to be a GIL ref). `String` copies the text out.
    if let Ok(text) = obj.extract::<String>() {
        if text == "all" {
            return Ok(SpecialMode::All);
        }
        return Err(PyValueError::new_err(
            "allowed_special string must be \"all\"",
        ));
    }
    let items: HashSet<String> = obj.extract().map_err(|_| {
        PyValueError::new_err("allowed_special must be \"all\" or a collection of strings")
    })?;
    Ok(SpecialMode::Only(items))
}

fn encode_with(
    tok: &RankTokenizer,
    text: &str,
    mode: &SpecialMode,
) -> Result<Vec<u32>, TokenizerError> {
    match mode {
        SpecialMode::All => tok.encode(text),
        SpecialMode::Only(allowed) => {
            let set: HashSet<&str> = allowed.iter().map(String::as_str).collect();
            tok.encode_with_special_tokens(text, &set)
        }
    }
}

fn default_vocab_path() -> PyResult<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    let Some(home) = home else {
        return Err(PyValueError::new_err(
            "HOME is not set; pass the path to cl100k_base.tiktoken",
        ));
    };
    Ok(PathBuf::from(home).join(".cache/tokenizer/cl100k_base.tiktoken"))
}

fn vocab_path(path: Option<String>) -> PyResult<PathBuf> {
    let path = match path {
        Some(path) => PathBuf::from(path),
        None => default_vocab_path()?,
    };
    if !path.is_file() {
        return Err(PyValueError::new_err(format!(
            "cl100k vocab not found at {}; run scripts/fetch-cl100k.sh",
            path.display()
        )));
    }
    Ok(path)
}

/// Local wrapper so `From<..> for PyErr` is not an orphan impl.
///
/// `TokenizerError` lives in the core crate and `PyErr` lives in PyO3.
/// `impl From<TokenizerError> for PyErr` is E0117: neither type is local.
/// `TokErr` is local, so `From<TokenizerError> for TokErr` and
/// `From<TokErr> for PyErr` are both legal. `?` after a `.map_err(TokErr)`
/// is the second conversion, and it runs only once the GIL is held again.
/// Every variant becomes `ValueError`. The message is `TokenizerError`'s
/// `Display` (`disallowed special token "<|endoftext|>"`, …).
struct TokErr(TokenizerError);

impl From<TokenizerError> for TokErr {
    fn from(err: TokenizerError) -> Self {
        Self(err)
    }
}

impl From<TokErr> for PyErr {
    fn from(err: TokErr) -> Self {
        PyValueError::new_err(err.0.to_string())
    }
}

/// cl100k_base encoder.
///
/// `frozen` turns off PyO3's runtime borrow flag. The value has to be `Sync`
/// anyway: free-threaded Python shares `#[pyclass]` objects across threads,
/// and rayon will call `&self` methods from worker threads. The vocab sits
/// behind [`Arc`] so those workers share one `RankTokenizer` without cloning
/// the hash maps.
#[pyclass(frozen, module = "mahad_tok")]
struct Encoding {
    inner: Arc<RankTokenizer>,
}

#[pymethods]
impl Encoding {
    /// Load `cl100k_base.tiktoken`.
    ///
    /// `path` defaults to `~/.cache/tokenizer/cl100k_base.tiktoken`.
    #[classmethod]
    #[pyo3(signature = (path=None))]
    fn cl100k_base(_cls: &Bound<'_, PyType>, path: Option<String>) -> PyResult<Self> {
        let path = vocab_path(path)?;
        let path_str = path
            .to_str()
            .ok_or_else(|| PyValueError::new_err("vocab path is not utf-8"))?;
        let tok = RankTokenizer::load_cl100k(path_str).map_err(TokErr)?;
        Ok(Self {
            inner: Arc::new(tok),
        })
    }

    /// Encode `text`.
    ///
    /// `allowed_special` defaults to an empty set, matching tiktoken: a
    /// registered special in `text` raises `ValueError`. Pass `"all"` to
    /// allow every special, or a set of token strings.
    ///
    /// `text` is borrowed from the Python `str` (no copy when it is already
    /// UTF-8). [`Python::detach`] runs the encode without the GIL. The
    /// closure does not touch a Python object: `RankTokenizer::encode` only
    /// reads the shared vocab.
    #[pyo3(signature = (text, *, allowed_special=None))]
    fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<u32>> {
        let mode = special_mode(allowed_special)?;
        let inner = Arc::clone(&self.inner);
        Ok(py
            .detach(move || encode_with(&inner, text, &mode))
            .map_err(TokErr)?)
    }

    /// Decode token ids to a string.
    ///
    /// An id that is not in the vocab raises `ValueError` (`UnknownToken`).
    /// Invalid UTF-8 does too.
    fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        Ok(py.detach(move || inner.decode(&ids)).map_err(TokErr)?)
    }

    /// Encode each string, in order.
    ///
    /// `num_threads` defaults to the machine's parallelism. `1` is a plain
    /// loop on this thread. Any larger value runs [`rayon`] on a cached pool
    /// inside [`Python::detach`], so other Python threads can run and the
    /// workers are not stuck waiting for the GIL.
    ///
    /// Inputs are copied to `Vec<String>` before the detach. The closure is
    /// `Send` and owns everything it reads. Worker threads only call
    /// `&RankTokenizer` methods. `RankTokenizer` is `Sync` (`HashMap`, `Vec`,
    /// and `fancy_regex::Regex` are `Sync`).
    #[pyo3(signature = (texts, *, num_threads=None, allowed_special=None))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        num_threads: Option<usize>,
        allowed_special: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let threads = thread_count(num_threads)?;
        let mode = special_mode(allowed_special)?;
        let inner = Arc::clone(&self.inner);
        // One string has nothing to split across workers. Skip the pool.
        let pool = if threads == 1 || texts.len() <= 1 {
            None
        } else {
            Some(cached_pool(threads)?)
        };
        Ok(py
            .detach(move || -> Result<Vec<Vec<u32>>, TokenizerError> {
                match &pool {
                    None => texts
                        .iter()
                        .map(|text| encode_with(&inner, text, &mode))
                        .collect(),
                    Some(pool) => pool.install(|| {
                        texts
                            .par_iter()
                            .map(|text| encode_with(&inner, text, &mode))
                            .collect()
                    }),
                }
            })
            .map_err(TokErr)?)
    }
}

/// Native cl100k encoder. Import `Encoding` from this module.
#[pymodule]
fn mahad_tok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Encoding>()?;
    Ok(())
}
