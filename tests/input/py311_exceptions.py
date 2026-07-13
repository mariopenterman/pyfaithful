def safe(fn, cleanup):
    try:
        value = fn()
    except KeyError as exc:
        value = exc.args
    except (ValueError, TypeError):
        value = None
    else:
        value = value + 1
    finally:
        cleanup()
    return value
