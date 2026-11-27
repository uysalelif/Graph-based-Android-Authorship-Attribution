# Source code-based (smali), permission and library feature extraction.
# The feature definitions and the library / permission lists follow Aydogan and Sen (2024),
# "Android Authorship Attribution Using Source Code-Based Features," IEEE Access,
# doi: 10.1109/ACCESS.2024.3351945; code and lists: https://github.com/emreaydogan/SourceCodeBasedAAA

import glob
import os
import re
import shutil
import subprocess
import sys

from lxml import etree


def getLibraryFeature(dec_path):
    library_feature = {}
    libraries = []

    with open(os.getcwd() + "/Data/library/library_list.txt", 'r+') as f:
        lineno = 0
        for line in f:
            lineno += 1
            libraries.append(line[:-1].strip())

    for lib in libraries:
        library_feature[lib] = 0

    for root, subs, files in os.walk("%s" % dec_path + "/smali"):
        for lib in libraries:
            if root.split('smali/')[-1].__eq__(lib):
                library_feature[lib] = 1

    return library_feature


def parsePermission(dec_path):
    tree = etree.ElementTree(file="%s/AndroidManifest.xml" % dec_path)
    root = tree.getroot()
    ns = root.nsmap['android']
    ns = "{%s}" % ns
    i = 0

    permission_feature = {}
    with open(os.getcwd() + "/Data/permission/permission_list_158.txt", "r") as permission_file:
        for perm in permission_file:
            permission_feature[perm.strip()] = 0
        # permissions = permission_file.readlines()
        # for perm in permissions:
        #     permission_feature[perm.strip()] = 0

    for usedPer in root.iter("uses-permission"):
        perName = (usedPer.attrib[ns + 'name']).lower()
        if perName in permission_feature:
            permission_feature[perName] = 1
            i += 1

    return permission_feature


def parseSmali(smali_files):
    smali_feature = {'avgCharPerLine': 0,
                     'avgCharPerLocalVar': 0,
                     'avgCharPerGlobalVar': 0,
                     'avgCharPerFuncName': 0,
                     'avgCharPerFuncParamName': 0,
                     'avgLinePerClass': 0,
                     'avgLinePerFunc': 0,
                     'avgFuncPerClass': 0,
                     'ratioGlobalVarToAllCodes': 0,
                     'ratioLocalVarToAllCodes': 0,
                     'ratioVarToAllCodes': 0,
                     'ratioIfToAllCodes': 0,
                     'ratioInvokeToAllCodes': 0,
                     'ratioIntToAllFunc': 0,
                     'ratioVoidToAllFunc': 0,
                     'ratioUpperToAllFunc': 0,
                     'ratioUpperToAllVar': 0,
                     'ratioMoveToAllCodes': 0,
                     }

    number_of_line = 0
    number_of_func_line = 0
    number_of_local_var = 0
    number_of_global_var = 0
    number_of_upper_var = 0
    number_of_func = 0
    number_of_upper_func = 0
    number_of_param = 0
    number_of_if = 0
    number_of_invoke = 0
    number_of_move = 0
    number_of_I_return_type = 0
    number_of_V_return_type = 0
    codes = []
    try:
        for smali_file in smali_files:
            with open(smali_file, "r") as file:
                copy = False
                for line in file:
                    code = line.strip()
                    if code != "" and not code.startswith("#"):
                        codes.append(code)
                        number_of_line += 1
                        if code.startswith(".method"):  # average Char per function name
                            copy = True
                            return_type = code.split(" ")[-1].split(")")[1]
                            if return_type.__eq__("I"):
                                number_of_I_return_type += 1
                            elif return_type.__eq__("V"):
                                number_of_V_return_type += 1
                            function_name = code.split(" ")[-1].split("(")[0]
                            match = re.search("\<\\w*init\>", function_name)
                            if not match:
                                smali_feature["avgCharPerFuncName"] = float(
                                    smali_feature["avgCharPerFuncName"] * number_of_func + len(function_name)) / (
                                                                                  number_of_func + 1)
                                if function_name.isupper():
                                    number_of_upper_func += 1
                                number_of_func += 1
                            continue
                        elif ".end method" in code:
                            copy = False
                            continue
                        elif code.startswith(".param"):  # average number of Function Parameter Name
                            param_name = code.split(",")[-1].split("#")[0].strip()[1:-1]
                            match = re.search("this\$", param_name)
                            if not match:
                                smali_feature["avgCharPerFuncParamName"] = float(
                                    smali_feature["avgCharPerFuncParamName"] * number_of_param + len(param_name)) / (
                                                                                       number_of_param + 1)
                                number_of_param += 1
                            continue
                        elif code.startswith(".field"):  # avg char per global var
                            glob = code.split(":")[0].split(" ")[-1]
                            smali_feature["avgCharPerGlobalVar"] = float(
                                smali_feature["avgCharPerGlobalVar"] * number_of_global_var + len(glob)) / (
                                                                               number_of_global_var + 1)
                            if glob.isupper():
                                number_of_upper_var += 1
                            number_of_global_var += 1
                            continue
                        elif code.startswith(".local"):  # avg char per local var
                            spl = code.split(",")
                            if len(spl) > 1:
                                local = spl[1].split(":")[0][2:-1]
                                smali_feature["avgCharPerLocalVar"] = float(
                                    smali_feature["avgCharPerLocalVar"] * number_of_local_var + len(local)) / (
                                                                                  number_of_local_var + 1)
                                if local.isupper():
                                    number_of_upper_var += 1
                                number_of_local_var += 1
                            continue
                        elif code.startswith("if-"):
                            number_of_if += 1
                            continue
                        elif code.startswith("invoke-"):
                            number_of_invoke += 1
                            continue
                        elif code.startswith("move-"):
                            number_of_move += 1
                            continue
                        if copy:
                            number_of_func_line += 1
                            continue

        if number_of_line != 0:
            smali_feature["avgCharPerLine"] = float(codes.__len__()) / number_of_line
            smali_feature["ratioGlobalVarToAllCodes"] = float(number_of_global_var) / number_of_line
            smali_feature["ratioLocalVarToAllCodes"] = float(number_of_local_var) / number_of_line
            smali_feature["ratioVarToAllCodes"] = float(number_of_local_var + number_of_global_var) / number_of_line
            smali_feature["ratioIfToAllCodes"] = float(number_of_if) / number_of_line
            smali_feature["ratioInvokeToAllCodes"] = float(number_of_invoke) / number_of_line
            smali_feature["ratioMoveToAllCodes"] = float(number_of_move) / number_of_line

        if number_of_func != 0:
            smali_feature["avgLinePerFunc"] = float(number_of_func_line) / number_of_func
            smali_feature["ratioIntToAllFunc"] = float(number_of_I_return_type) / number_of_func
            smali_feature["ratioVoidToAllFunc"] = float(number_of_V_return_type) / number_of_func
            smali_feature["ratioUpperToAllFunc"] = float(number_of_upper_func) / number_of_func

        if len(smali_files) != 0:
            smali_feature["avgLinePerClass"] = float(number_of_line) / len(smali_files)
            smali_feature["avgFuncPerClass"] = float(number_of_func) / len(smali_files)

        if (number_of_local_var + number_of_global_var) != 0:
            smali_feature["ratioUpperToAllVar"] = float(number_of_upper_var) / (
                        number_of_local_var + number_of_global_var)

    except:
        err = str(sys.exc_info()[0]) + str(sys.exc_info()[1])
        print(err)

    return smali_feature


def get_smali_features(dec_path):
    smali_folder_path = dec_path + "/smali/"

    smali_files = []
    for root, subs, files in os.walk(smali_folder_path):
        for filename in files:
            smali_files.append(os.path.join(root, filename))

    # for s in glob.glob(smali_folder_path + "/**/*.smali", recursive=True):
    #     smali_files.append(s)

    smali_feats = parseSmali(smali_files)
    sfs = sorted(smali_feats.items(), key=lambda k: k[0])
    smaliFeature = []
    for sf in sfs:
        smaliFeature.append(float(sf[1]))
    #smali_feats = list(smali_feats.values())
    #print("smali features are extracted!")
    #print(f"smali features: {smali_feats}")

    return smaliFeature

