"""pytest 없이도 돌아가는 아주 작은 테스트 도우미."""

from contextlib import contextmanager


@contextmanager
def expect_raises(exc_type, match=None):
    """with 블록 안에서 exc_type 예외가 나야 통과. match 가 있으면 메시지에 포함돼야 한다."""
    try:
        yield
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"예외 메시지에 {match!r} 이(가) 없습니다: {exc}") from exc
    else:
        raise AssertionError(f"{exc_type.__name__} 예외가 발생해야 합니다.")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol
