from imas.wrangler import wrangle
from .fetchers import simple_load


def export_to_ids(fields, shot, **kwargs):
    flat = simple_load(fields, shot, **kwargs)
    return wrangle(flat, "3.41.0")


if __name__ == "__main__":
    print(export_to_ids('equilibrium.time', 200000).time)