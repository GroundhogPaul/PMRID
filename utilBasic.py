import sys
import os

sPathFolderUtilVrf = "D:/users/xiaoyaopan/PxyAI/DataSet/Vrf"
assert os.path.exists(sPathFolderUtilVrf), f"Folder does not exist: {sPathFolderUtilVrf}"
sys.path.append(sPathFolderUtilVrf)
import utilVrf
