""" Script for analysis of the individual pricing """
from dotmap import DotMap


def calculate_max_discount(
        databank: DotMap
) -> DotMap:
    ns_utilities = {t[0]: t[1]['u'] for t in databank['exmas']['requests'].iterrows()}

