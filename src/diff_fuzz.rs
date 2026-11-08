//! Differential tests for cl100k full-text encode.
//!
//! `prop_round_trip_bytes`: encode then decode returns the same `String`.
//! `prop_matches_tiktoken_rs`: the ids match `tiktoken_rs::cl100k_base`
//! (`encode_with_special_tokens`). Python tiktoken is the other oracle; the
//! cross-check script is `scripts/cross_check_tiktoken.py`. If this crate and
//! tiktoken-rs ever disagree, trust Python.
//!
//! Both properties draw from [`biased_text`]: `prop_oneof!` over whitespace,
//! digit runs, apostrophe contractions, emoji, the five cl100k specials, and
//! arbitrary Unicode scalars. Earlier arms are simpler, which is the order
//! proptest shrinks toward.
//!
//! Case count is 10_000 when `PROPTEST_CASES` is unset. proptest 1.12's
//! `Config::default` already applies that variable (its own fallback is 256).
//! Setting `PROPTEST_CASES` to a smaller `u32` shortens a slow debug or CI
//! run without editing this file. The same variable is what
//! `.github/workflows/rust.yml` documents. Fuzzing is not part of `cargo test`.

use std::sync::OnceLock;

use proptest::prelude::*;

use crate::RankTokenizer;
use crate::rank::ensure_cl100k_file;

const SPECIALS: &[&str] = &[
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
];

const EMOJI: &[&str] = &["🦀", "😀", "👨‍👩‍👧‍👦", "é", "你", "日", "a\u{0301}", "\u{200d}"];

fn cl100k() -> &'static RankTokenizer {
    static MODEL: OnceLock<RankTokenizer> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = ensure_cl100k_file();
        RankTokenizer::load_cl100k(path.to_str().expect("cache path is utf-8"))
            .unwrap_or_else(|err| panic!("load_cl100k({}): {err}", path.display()))
    })
}

fn oracle() -> &'static tiktoken_rs::CoreBPE {
    static ORACLE: OnceLock<tiktoken_rs::CoreBPE> = OnceLock::new();
    ORACLE.get_or_init(|| {
        tiktoken_rs::cl100k_base().unwrap_or_else(|err| panic!("tiktoken-rs cl100k_base: {err}"))
    })
}

/// 10_000 cases unless `PROPTEST_CASES` is set. `default()` reads the
/// variable; overwriting `cases` afterwards would ignore it.
fn proptest_config() -> ProptestConfig {
    let mut config = ProptestConfig::default();
    if std::env::var("PROPTEST_CASES").is_err() {
        config.cases = 10_000;
    }
    // A failing case is printed by the harness. Do not also write
    // `proptest-regressions/` into the tree.
    config.failure_persistence = None;
    config
}

fn regex_string(pat: &'static str) -> impl Strategy<Value = String> {
    // `string_regex` is `Result`. The `&str: Strategy` impl unwraps the same
    // call later, which hides a bad pattern inside the first generated case.
    proptest::string::string_regex(pat)
        .unwrap_or_else(|err| panic!("proptest regex {pat:?}: {err}"))
}

/// One biased piece. Weights are relative, not probabilities.
fn biased_chunk() -> impl Strategy<Value = String> {
    prop_oneof![
        1 => Just(String::new()),
        4 => regex_string("[ \t\r\n]{1,12}"),
        4 => regex_string("[0-9]{1,18}"),
        4 => regex_string("[A-Za-z]{1,10}'(s|S|t|T|d|D|m|M|ll|LL|ve|VE|re|RE)"),
        3 => prop::sample::select(EMOJI).prop_map(str::to_owned),
        3 => prop::sample::select(SPECIALS).prop_map(str::to_owned),
        // `any::<char>()` is any Unicode scalar, including controls. The
        // closure return type is named so `collect` is `String` and not
        // another `FromIterator<char>` target.
        2 => prop::collection::vec(any::<char>(), 1..24)
            .prop_map(|chars| -> String { chars.into_iter().collect() }),
    ]
}

/// Several pieces concatenated, so a special or a digit run can sit in the
/// middle of other text. Length shrinks before the pieces do.
fn biased_text() -> impl Strategy<Value = String> {
    prop::collection::vec(biased_chunk(), 0..6).prop_map(|parts| parts.concat())
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn prop_round_trip_bytes(text in biased_text()) {
        let tok = cl100k();
        let ids = tok
            .encode(&text)
            .unwrap_or_else(|err| panic!("encode {text:?}: {err}"));
        prop_assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    #[test]
    fn prop_matches_tiktoken_rs(text in biased_text()) {
        let tok = cl100k();
        let ours = tok
            .encode(&text)
            .unwrap_or_else(|err| panic!("encode {text:?}: {err}"));
        // `Rank` is `u32`, the same integer `encode` returns.
        let theirs = oracle().encode_with_special_tokens(&text);
        prop_assert_eq!(ours, theirs);
    }
}
