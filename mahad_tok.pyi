from collections.abc import Iterable, Sequence

class Encoding:
    @classmethod
    def cl100k_base(cls, path: str | None = None) -> Encoding: ...
    def encode(
        self,
        text: str,
        *,
        allowed_special: str | Iterable[str] | None = None,
    ) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        num_threads: int | None = None,
        allowed_special: str | Iterable[str] | None = None,
    ) -> list[list[int]]: ...
