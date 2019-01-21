import datetime


def gettime():
    """
    return a time str for now
    """
    now = datetime.datetime.now()
    date = "%s-%s-%s_%s-%s-%s" % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    return date
