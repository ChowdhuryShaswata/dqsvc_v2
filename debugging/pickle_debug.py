# put near your imports
try:
    import cloudpickle
except ImportError:
    cloudpickle = None

def diagnose_joblib_pickle(func, *args, **kwargs):
    """Check cloudpickle-ability of func, each arg, each kwarg, and the full call tuple."""
    if cloudpickle is None:
        raise RuntimeError("cloudpickle not available; pip install cloudpickle")
    report = {"function": True, "args": [], "kwargs": {}, "call_tuple": True, "error": None}

    try:
        cloudpickle.dumps(func)
    except Exception as e:
        report["function"] = False
        report["call_tuple"] = False
        report["error"] = ("function", e)
        return report

    for i, a in enumerate(args):
        try:
            cloudpickle.dumps(a)
            report["args"].append(True)
        except Exception as e:
            report["args"].append(False)
            if report["error"] is None:
                report["error"] = (f"args[{i}]", e)

    for k, v in kwargs.items():
        try:
            cloudpickle.dumps(v)
            report["kwargs"][k] = True
        except Exception as e:
            report["kwargs"][k] = False
            if report["error"] is None:
                report["error"] = (f"kwargs['{k}']", e)

    try:
        cloudpickle.dumps((func, args, kwargs))
    except Exception as e:
        report["call_tuple"] = False
        if report["error"] is None:
            report["error"] = ("call_tuple", e)

    return report
