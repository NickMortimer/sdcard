"""
File system utils
"""

import os
from math import  ceil
import psutil
            
def list_sdcards(format_type,maxcardsize=512):
    """
    Scan for SD cards.

    Args:
        format_type : type of format on the sdcard (exfat preffered)
        maxcardsize : select drives with less than the max in Gb
    """
    result =[]
    for i in psutil.disk_partitions():
        if i.fstype.lower()==format_type:
            p =psutil.disk_usage(i.mountpoint)
            if ceil(p.total/1000000000)<=maxcardsize:            
                result.append(i.mountpoint)
    return result