# -*- coding: utf-8 -*-
# Adapted from authfeature.py of AppAuth (https://github.com/buptkick/AppAuth;
# M. Xu et al., "AppAuth: Authorship Attribution for Android App Clones," IEEE Access, 2019).
# The source code-based, permission and library features (spl_feature_extraction.py,
# following Aydogan and Sen, 2024) were added for this study.
"""
Created on Tue Aug 14 17:50:58 2018

@author: softsec-zcp
"""

import sys, time
from _settings import *
import literadar
import dex_parser
import numpy as np
import csv
import multiprocessing
import json
import shutil
import spl_feature_extraction as spl

from lxml import etree

remove_tpl = True

f_log = open("log_genome_appauth_spl.txt", "w")

SENSITIVE_PERMISSIONS = [
    "android.permission.WRITE_CONTACTS",
    "android.permission.GET_ACCOUNTS",
    "android.permission.READ_CONTACTS",
    "android.permission.READ_CALL_LOG",
    "android.permission.READ_PHONE_STATE",
    "android.permission.CALL_PHONE",
    "android.permission.WRITE_CALL_LOG",
    "android.permission.USE_SIP",
    "android.permission.PROCESS_OUTGOING_CALLS",
    "com.android.voicemail.permission.ADD_VOICEMAIL",
    "android.permission.READ_CALENDAR",
    "android.permission.WRITE_CALENDAR",
    "android.permission.CAMERA",
    "android.permission.BODY_SENSORS",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.RECORD_AUDIO",
    "android.permission.READ_SMS",
    "android.permission.RECEIVE_WAP_PUSH",
    "android.permission.RECEIVE_MMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.SEND_SMS",
    "android.permission.READ_CELL_BROADCASTS"]

# method functional opcode：
opIns = ["invoke", "return", "new", "move", "const", "monitor",
         "get", "put", "cmp", "if", "goto", "switch", "check", "throw",
         "instance", "array"]

# method mathematical opcode：
mathIns = ["neg-", "not-", "-to-", "add-", "sub-",
           "mul-", "div-", "rem-", "and-", "xor-", "or-", "shl-",
           "ushr-", "shr-"]

# method array opcode：
arrayIns = ["array-length", "new-array", "filled-new-array", "filled-new-array/range", "fill-array-data",
            "aget", "aget-wide", "aget-object", "aget-boolean", "aget-byte", "aget-char", "aget-short",
            "aput", "aput-wide", "aput-object", "aput-boolean", "aput-byte", "aput-char", "aput-short"]

constIns = ['const/4', 'const/16', 'const', 'const-wide/16', 'const-wide/32', 'const-wide']


def parseAndroidManifest(lrd):
    tree = etree.ElementTree(file="%s/AndroidManifest.xml" % lrd.dec_path)
    root = tree.getroot()
    ns = root.nsmap['android']
    ns = "{%s}" % ns

    usesPermissionList = []
    susesPermissionList = []
    usesFeatureList = []
    # filtering uses-permission label
    for usesPermission in root.iter("uses-permission"):
        usesPermissionName = usesPermission.attrib[ns + 'name']
        usesPermissionList.append(usesPermissionName)
        if usesPermissionName in SENSITIVE_PERMISSIONS:
            susesPermissionList.append(usesPermissionName)

    for usesFeature in root.iter("uses-feature"):
        for k, v in usesFeature.attrib.items():
            usesFeatureList.append(v)

    amFeature = {'usesPermissionNum': len(usesPermissionList),
                 'susesPermissionNum': len(susesPermissionList),
                 'usesFeatureNum': len(usesFeatureList),
                 'intentFilterNum': 0,
                 'metaDataNum': 0,
                 'activityNum': 0,
                 'receiverNum': 0,
                 'serviceNum': 0,
                 'providerNum': 0,
                 }

    for intentFilter in root.iter("intent-filter"):
        amFeature['intentFilterNum'] += 1

    for metaData in root.iter("meta-data"):
        amFeature['metaDataNum'] += 1

    for activity in root.iter("activity"):
        amFeature['activityNum'] += 1

    for receiver in root.iter("receiver"):
        amFeature['receiverNum'] += 1

    for service in root.iter("service"):
        amFeature['serviceNum'] += 1

    for provider in root.iter("provider"):
        amFeature['providerNum'] += 1

    return amFeature


def parseResources(lrd):
    resFeature = {'assetsNum': 0,
                  'libNum': 0, 'libSoNum': 0,
                  'resNum': 0, 'resXmlDirFileNum': 0,
                  'resDrawableDirNum': 0, 'resDrawableFileNum': 0,
                  'resValuesDirNum': 0, 'resValuesFileNum': 0,
                  'resLayoutDirNum': 0, 'resLayoutFileNum': 0,
                  }

    for mainDirFile in os.listdir(lrd.dec_path):
        if mainDirFile == "assets":
            for assetsFile in os.listdir(lrd.dec_path + "/assets"):
                resFeature['assetsNum'] += 1
        elif mainDirFile == "lib":
            for libFile in os.listdir(lrd.dec_path + "/lib"):
                resFeature['libNum'] += 1
                if libFile.endswith(".so"):
                    resFeature['libSoNum'] += 1
                libDir = lrd.dec_path + "/lib/" + libFile
                if os.path.isdir(libDir):
                    try:
                        for libFile1 in os.listdir(libDir):
                            if libFile1.endswith(".so"):
                                resFeature['libSoNum'] += 1
                    except:
                        pass
        elif mainDirFile == "res":
            for resFile in os.listdir(lrd.dec_path + "/res"):
                resFeature['resNum'] += 1
                if resFile == "xml":
                    for xmlFile in os.listdir(lrd.dec_path + "/res/" + resFile):
                        resFeature['resXmlDirFileNum'] += 1
                elif resFile.startswith("drawable"):
                    resFeature['resDrawableDirNum'] += 1
                    for drawableFile in os.listdir(lrd.dec_path + "/res/" + resFile):
                        resFeature['resDrawableFileNum'] += 1
                elif resFile.startswith("values"):
                    resFeature['resValuesDirNum'] += 1
                    for valuesFile in os.listdir(lrd.dec_path + '/res/' + resFile):
                        resFeature['resValuesFileNum'] += 1
                elif resFile.startswith("layout"):
                    resFeature['resLayoutDirNum'] += 1
                    for valuesFile in os.listdir(lrd.dec_path + '/res/' + resFile):
                        resFeature['resLayoutFileNum'] += 1

    return resFeature


def judgeTpl(packageName, res):
    isTpl = False
    # print("\n",res,"\n")
    for r in res:
        if packageName.startswith(r['Package']):
            isTpl = True
            break
    return isTpl


def parseMethod(dexMethod, dex_object):
    methodFeature = {'registersSize': 0, 'insSize': 0,
                     'outsSize': 0, 'triesSize': 0,
                     'debugInfoOff': 0, 'insnsSize': 0,
                     'opInsNum': 0, 'mathInsNum': 0,
                     'aInsNum': 0, 'aInsDict': {
            'naNum': 0, 'alNum': 0, 'fadNum': 0, 'fnaNum': 0, 'fnarNum': 0,
            'agNum': 0, 'apNum': 0, },
                     'nalNum': 0, 'nalDict': {},
                     }

    if dexMethod.dexCode == None:
        return None

    methodFeature['registersSize'] = dexMethod.dexCode.registersSize
    methodFeature['insSize'] = dexMethod.dexCode.insSize
    methodFeature['outsSize'] = dexMethod.dexCode.outsSize
    methodFeature['triesSize'] = dexMethod.dexCode.triesSize
    methodFeature['debugInfoOff'] = dexMethod.dexCode.debugInfoOff
    methodFeature['insnsSize'] = dexMethod.dexCode.insnsSize

    offset = 0
    insnsSize = dexMethod.dexCode.insnsSize * 4

    cdict = {}

    while offset < insnsSize:
        opcode = int(dexMethod.dexCode.insns[offset:offset + 2], 16)
        formatIns, _ = dex_parser.getOpCode(opcode)

        decodedInstruction = dex_parser.dexDecodeInstruction(dex_object, dexMethod.dexCode, offset)

        smaliCode = decodedInstruction.smaliCode
        if smaliCode == None:
            continue

        opFlag = False
        for ins in opIns:
            if ins in decodedInstruction.op:
                opFlag = True
                methodFeature['opInsNum'] += 1
                if decodedInstruction.op in constIns:
                    cdict[decodedInstruction.vA] = int(decodedInstruction.vB)
                elif decodedInstruction.op in arrayIns:
                    methodFeature['aInsNum'] += 1
                    if decodedInstruction.op == 'new-array':
                        methodFeature['aInsDict']['naNum'] += 1
                        try:
                            naLen = cdict[decodedInstruction.vA]
                            if naLen in methodFeature['nalDict'].keys():
                                methodFeature['nalDict'][naLen] += 1
                            else:
                                methodFeature['nalDict'][naLen] = 1
                            methodFeature['nalNum'] += 1
                        except:
                            pass
                    elif decodedInstruction.op == 'arrary-lengh':
                        methodFeature['aInsDict']['alNum'] += 1
                    elif decodedInstruction.op == 'fill-array-data':
                        methodFeature['aInsDict']['fadNum'] += 1
                    elif decodedInstruction.op == 'filled-new-array':
                        methodFeature['aInsDict']['fnaNum'] += 1
                    elif decodedInstruction.op == 'filled-new-array/range':
                        methodFeature['aInsDict']['fnarNum'] += 1
                    elif decodedInstruction.op.startswith('aget'):
                        methodFeature['aInsDict']['agNum'] += 1
                    else:
                        methodFeature['aInsDict']['apNum'] += 1
                break

        if not opFlag:
            methodFeature['mathInsNum'] += 1

        insns = dexMethod.dexCode.insns[decodedInstruction.offset:decodedInstruction.offset + decodedInstruction.length]
        offset += len(insns)

        if smaliCode == 'nop':
            break

    return methodFeature


def parseClass(dexClassDefObj, dex_object):
    classFeature = {'implementsNum': 0, 'annotationsNum': 0, 'sourceNum': 0, 'dataNum': 0,
                    'staticFieldsNum': 0, 'instanceFieldsNum': 0,
                    'directMethodsNum': 0, 'virtualMethodsNum': 0,
                    'accessFlags': 0,
                    'trycatMethodNum': 0, 'dbginfMethodNum': 0,
                    'opInsNum': 0, 'mathInsNum': 0,
                    'aInsDict': {}, 'nalDict': {},
                    }

    if dexClassDefObj.header is None:
        return None

    classFeature['staticFieldsNum'] = dexClassDefObj.header.staticFieldsSize
    classFeature['instanceFieldsNum'] = dexClassDefObj.header.instanceFieldsSize
    classFeature['directMethodsNum'] = dexClassDefObj.header.directMethodsSize
    classFeature['virtualMethodsNum'] = dexClassDefObj.header.virtualMethodsSize

    classFeature['annotationsNum'] = dexClassDefObj.annotationsOff
    classFeature['implementsNum'] = dexClassDefObj.interfaceOff
    classFeature['sourceNum'] = dexClassDefObj.sourceFieldIdx
    classFeature['dataNum'] = dexClassDefObj.classDataOff
    classFeature['accessFlags'] = dexClassDefObj.accessFlags

    methodFeatureList = []

    for k in range(len(dexClassDefObj.directMethods)):
        methodFeature = parseMethod(dexClassDefObj.directMethods[k], dex_object)
        if methodFeature is not None:
            methodFeatureList.append(methodFeature)

    for k in range(len(dexClassDefObj.virtualMethods)):
        methodFeature = parseMethod(dexClassDefObj.virtualMethods[k], dex_object)
        if methodFeature is not None:
            methodFeatureList.append(methodFeature)

    for mf in methodFeatureList:
        if mf['debugInfoOff'] > 0:
            classFeature['dbginfMethodNum'] += 1

        if mf['triesSize'] > 0:
            classFeature['trycatMethodNum'] += 1

        classFeature['opInsNum'] += mf['opInsNum']
        classFeature['mathInsNum'] += mf['mathInsNum']

        for k, v in mf['aInsDict'].items():
            if k in classFeature['aInsDict'].keys():
                classFeature['aInsDict'][k] += v
            else:
                classFeature['aInsDict'][k] = v

        for k, v in mf['nalDict'].items():
            if k in classFeature['nalDict'].keys():
                classFeature['nalDict'][k] += v
            else:
                classFeature['nalDict'][k] = v

    #    print classFeature
    return classFeature


def parseDexSmali(lrd, res):
    dexFeature = {  # 'ownClassRatio':0.0,
        'absClassRatio': 0.0,
        'antClassRatio': 0.0, 'infClassRatio': 0.0,
        'drtMethodRatio': 0.0, 'vtlMethodRatio': 0.0,
        'trcMethodRatio': 0.0, 'dbiMethodRatio': 0.0,
        'statFieldRatio': 0.0,
        'mathInsRatio': 0.0,
        'arrayAverage': 0.0, 'arrayMedian': 0.0, 'arrayStd': 0.0,
        'agetRatio': 0.0, 'aInsDict': {},
        'naConstRatio': 0.0, 'nalDict': {},
    }

    ownClassNum = 0
    tplClassNum = 0
    allClassNum = 0
    absClassNum = 0
    parClassNum = 0

    datClassNum = 0
    antClassNum = 0
    infClassNum = 0
    srcClassNum = 0

    ownMethodNum = 0
    drtMethodNum = 0
    vtlMethodNum = 0
    trcMethodNum = 0
    dbiMethodNum = 0

    ownFieldNum = 0
    statFieldNum = 0
    instFieldNum = 0

    ownInsNum = 0
    opInsNum = 0
    mathInsNum = 0

    # print(lrd.dex_objects,"\n")

    for dex_object in lrd.dex_objects:
        allClassNum += len(dex_object.dexClassDefList)
        for index in range(len(dex_object.dexClassDefList)):
            dexClassDefObj = dex_object.dexClassDefList[index]
            #            print dex_object.getDexTypeId(dexClassDefObj.classIdx)
            # print(dex_object.getDexTypeId(dexClassDefObj.classIdx))
            if not judgeTpl(dex_object.getDexTypeId(dexClassDefObj.classIdx), res):
                #                print '    #%s~%s' % (hex(dexClassDefObj.offset), hex(dexClassDefObj.offset + dexClassDefObj.length))
                #                print '    DexClassDef[%d]:\t' % index
                #                print '    DexClassDef[%d]->classIdx\t= %s\t#%s' % (index, hex(dexClassDefObj.classIdx), dex_object.getDexTypeId(dexClassDefObj.classIdx))
                #                print '    DexClassDef[%d]->accessFlags\t= %s' % (index, hex(dexClassDefObj.accessFlags) )
                #                print '    DexClassDef[%d]->superclassIdx\t= %s\t#%s' % (index, hex(dexClassDefObj.superclassIdx), dex_object.getDexTypeId(dexClassDefObj.superclassIdx))
                #                print '    DexClassDef[%d]->interfaceOff\t= %s' % (index, hex(dexClassDefObj.interfaceOff))
                cf = parseClass(dexClassDefObj, dex_object)
                ownClassNum += 1
                if cf is not None:
                    parClassNum += 1

                    if cf['annotationsNum'] > 0:
                        antClassNum += 1

                    if cf['implementsNum'] > 0:
                        infClassNum += 1

                    if cf['sourceNum'] > 0:
                        srcClassNum += 1

                    if cf['dataNum'] > 0:
                        datClassNum += 1

                    drtMethodNum += cf['directMethodsNum']
                    vtlMethodNum += cf['virtualMethodsNum']
                    ownMethodNum += (drtMethodNum + vtlMethodNum)

                    trcMethodNum += cf['trycatMethodNum']
                    dbiMethodNum += cf['dbginfMethodNum']

                    statFieldNum += cf['staticFieldsNum']
                    instFieldNum += cf['instanceFieldsNum']
                    ownFieldNum += (statFieldNum + instFieldNum)

                    opInsNum += cf['opInsNum']
                    mathInsNum += cf['mathInsNum']
                    ownInsNum += (opInsNum + mathInsNum)

                    for k, v in cf['aInsDict'].items():
                        if k in dexFeature['aInsDict'].keys():
                            dexFeature['aInsDict'][k] += v
                        else:
                            dexFeature['aInsDict'][k] = v

                    for k, v in cf['nalDict'].items():
                        if k in dexFeature['nalDict'].keys():
                            dexFeature['nalDict'][k] += v
                        else:
                            dexFeature['nalDict'][k] = v
                else:
                    absClassNum += 1
            else:
                tplClassNum += 1

    arrayLens = np.array([])
    nalNum = 0

    for k, v in dexFeature['nalDict'].items():
        nalNum += v
        if k < 0x70000000:
            for i in range(v):
                t = np.array([k])
                arrayLens = np.hstack((arrayLens, t))

    if dexFeature['aInsDict']['naNum'] > 0:
        dexFeature['naConstRatio'] = float(nalNum) / dexFeature['aInsDict']['naNum']

    dexFeature['arrayAverage'] = np.average(arrayLens)
    dexFeature['arrayMedian'] = np.median(arrayLens)
    dexFeature['arrayStd'] = np.std(arrayLens)

    agpNum = dexFeature['aInsDict']['agNum'] + dexFeature['aInsDict']['apNum']
    dexFeature['agetRatio'] = float(dexFeature['aInsDict']['agNum']) / agpNum

    #    dexFeature['ownClassRatio'] = float(ownClassNum) / allClassNum

    dexFeature['absClassRatio'] = float(absClassNum) / ownClassNum

    dexFeature['antClassRatio'] = float(antClassNum) / parClassNum
    dexFeature['infClassRatio'] = float(infClassNum) / parClassNum

    dexFeature['drtMethodRatio'] = float(drtMethodNum) / ownMethodNum
    dexFeature['vtlMethodRatio'] = float(vtlMethodNum) / ownMethodNum

    dexFeature['trcMethodRatio'] = float(trcMethodNum) / ownMethodNum
    dexFeature['dbiMethodRatio'] = float(dbiMethodNum) / ownMethodNum

    dexFeature['statFieldRatio'] = float(statFieldNum) / ownFieldNum

    dexFeature['mathInsRatio'] = float(mathInsNum) / ownInsNum

    return dexFeature


def parseDexSmaliAll(lrd, res):
    dexFeature = {  # 'ownClassRatio':0.0,
        'absClassRatio': 0.0,
        'antClassRatio': 0.0, 'infClassRatio': 0.0,
        'drtMethodRatio': 0.0, 'vtlMethodRatio': 0.0,
        'trcMethodRatio': 0.0, 'dbiMethodRatio': 0.0,
        'statFieldRatio': 0.0,
        'mathInsRatio': 0.0,
        'arrayAverage': 0.0, 'arrayMedian': 0.0, 'arrayStd': 0.0,
        'agetRatio': 0.0, 'aInsDict': {},
        'naConstRatio': 0.0, 'nalDict': {},
    }

    ownClassNum = 0
    #    tplClassNum = 0
    allClassNum = 0
    absClassNum = 0
    parClassNum = 0

    datClassNum = 0
    antClassNum = 0
    infClassNum = 0
    srcClassNum = 0

    ownMethodNum = 0
    drtMethodNum = 0
    vtlMethodNum = 0
    trcMethodNum = 0
    dbiMethodNum = 0

    ownFieldNum = 0
    statFieldNum = 0
    instFieldNum = 0

    ownInsNum = 0
    opInsNum = 0
    mathInsNum = 0

    for dex_object in lrd.dex_objects:
        allClassNum += len(dex_object.dexClassDefList)
        for index in range(len(dex_object.dexClassDefList)):
            dexClassDefObj = dex_object.dexClassDefList[index]
            cf = parseClass(dexClassDefObj, dex_object)
            ownClassNum += 1
            if cf is not None:
                parClassNum += 1

                if cf['annotationsNum'] > 0:
                    antClassNum += 1

                if cf['implementsNum'] > 0:
                    infClassNum += 1

                if cf['sourceNum'] > 0:
                    srcClassNum += 1

                if cf['dataNum'] > 0:
                    datClassNum += 1

                drtMethodNum += cf['directMethodsNum']
                vtlMethodNum += cf['virtualMethodsNum']
                ownMethodNum += (drtMethodNum + vtlMethodNum)

                trcMethodNum += cf['trycatMethodNum']
                dbiMethodNum += cf['dbginfMethodNum']

                statFieldNum += cf['staticFieldsNum']
                instFieldNum += cf['instanceFieldsNum']
                ownFieldNum += (statFieldNum + instFieldNum)

                opInsNum += cf['opInsNum']
                mathInsNum += cf['mathInsNum']
                ownInsNum += (opInsNum + mathInsNum)

                for k, v in cf['aInsDict'].items():
                    if k in dexFeature['aInsDict'].keys():
                        dexFeature['aInsDict'][k] += v
                    else:
                        dexFeature['aInsDict'][k] = v

                for k, v in cf['nalDict'].items():
                    if k in dexFeature['nalDict'].keys():
                        dexFeature['nalDict'][k] += v
                    else:
                        dexFeature['nalDict'][k] = v
            else:
                absClassNum += 1

    arrayLens = np.array([])
    nalNum = 0

    for k, v in dexFeature['nalDict'].items():
        nalNum += v
        if k < 0x70000000:
            for i in range(v):
                t = np.array([k])
                arrayLens = np.hstack((arrayLens, t))

    if dexFeature['aInsDict']['naNum'] > 0:
        dexFeature['naConstRatio'] = float(nalNum) / dexFeature['aInsDict']['naNum']

    dexFeature['arrayAverage'] = np.average(arrayLens)
    dexFeature['arrayMedian'] = np.median(arrayLens)
    dexFeature['arrayStd'] = np.std(arrayLens)

    agpNum = dexFeature['aInsDict']['agNum'] + dexFeature['aInsDict']['apNum']
    dexFeature['agetRatio'] = float(dexFeature['aInsDict']['agNum']) / agpNum

    #    dexFeature['ownClassRatio'] = float(ownClassNum) / allClassNum

    dexFeature['absClassRatio'] = float(absClassNum) / ownClassNum

    dexFeature['antClassRatio'] = float(antClassNum) / parClassNum
    dexFeature['infClassRatio'] = float(infClassNum) / parClassNum

    dexFeature['drtMethodRatio'] = float(drtMethodNum) / ownMethodNum
    dexFeature['vtlMethodRatio'] = float(vtlMethodNum) / ownMethodNum

    dexFeature['trcMethodRatio'] = float(trcMethodNum) / ownMethodNum
    dexFeature['dbiMethodRatio'] = float(dbiMethodNum) / ownMethodNum

    dexFeature['statFieldRatio'] = float(statFieldNum) / ownFieldNum

    dexFeature['mathInsRatio'] = float(mathInsNum) / ownInsNum

    return dexFeature


def get_appauth_spl_features(lrd ,extracted_path):
    # apk_file = apk_file.rstrip()
    # lrd = literadar.LibRadarLite(apk_file)
    res = lrd.compare()

    apkFeature = []

    vectorIndex = 0

    try:
        af = parseAndroidManifest(lrd)
        # print("AndroidManifest feats:", af)
        afs = sorted(af.items(), key=lambda k: k[0])
        for a in afs:
            vectorIndex += 1
            apkFeature.append(float(a[1]))
    except:
        f_log.write("Error. \n")
        f_log.write(str(extracted_path)+"\n")
        f_log.write("apkFeature configuration\n\n")
        apkFeature.extend([np.nan]*9)

    try:
        rf = parseResources(lrd)
        # print("Resources feats:", rf)
        rfs = sorted(rf.items(), key=lambda k: k[0])
        for r in rfs:
            vectorIndex += 1
            apkFeature.append(float(r[1]))
    except:
        f_log.write("Error.\n")
        f_log.write(str(extracted_path)+"\n")
        f_log.write("apkFeature resource\n\n")
        apkFeature.extend([np.nan]*11)

    try:
        if remove_tpl:
            #   excluding TPL features
            df = parseDexSmali(lrd, res)
            # print("Dex Smali feats:",df)
        else:
            #   including TPL features
            df = parseDexSmaliAll(lrd, res)
            # print("Dex Smali feats (including):",df)

        dfs = sorted(df.items(), key=lambda k: k[0])
        for d in dfs:
            if type(d[1]) == int or type(d[1]) == float or type(d[1]) == np.float64:
                vectorIndex += 1
                apkFeature.append(float(d[1]))
    except:
        f_log.write("Error.\n")
        f_log.write(str(extracted_path)+"\n")
        f_log.write("apkFeature dex code\n\n")
        apkFeature.extend([np.nan]*14)

    ########################################
    # OTHER FEATURES
    # get smali features
    try:
        smali_feats = spl.get_smali_features(extracted_path)
        #print("\nsmali features are extracted!")
        #print("smali feats:", smali_feats)
    except:
        f_log.write("Error.\n")
        f_log.write(str(extracted_path)+"\n")
        f_log.write("smali_feats\n\n")
        smali_feats = [np.nan]*18

    # get permission features
    try:
        permission_feats = spl.parsePermission(extracted_path)
        pfs = sorted(permission_feats.items(), key=lambda k: k[0])
        permissionFeature=[]
        for perF in pfs:
            permissionFeature.append(float(perF[1]))
        #permission_feats = list(permission_feats.values())
        #print("\npermission features are extracted!")
        #print("permission features:", permissionFeature)
    except:
        f_log.write("Error.\n")
        f_log.write(str(extracted_path)+"\n")
        f_log.write("permissionFeature\n\n")
        permissionFeature = [np.nan]*158

    # get library features
    try:
        library_feats = spl.getLibraryFeature(extracted_path)
        lfs = sorted(library_feats.items(), key=lambda k: k[0])
        libraryFeature=[]
        for libF in lfs:
            libraryFeature.append(float(libF[1]))
        #library_feats = list(library_feats.values())
        #print("\nlibrary features are extracted!")
        #print("library features:", libraryFeature)
    except:
        f_log.write("Error.\n")
        f_log.write(str(extracted_path)+"\n")
        f_log.write("libraryFeature\n\n")
        libraryFeature = [np.nan]*500

    ############################################

    lrd.__close__()

    #print(apkFeature+smali_feats+permissionFeature+libraryFeature)
    return apkFeature+smali_feats+permissionFeature+libraryFeature


authfeats_file = open("genome_appauth_spl_features.json", "w")
authfeats_dict = {}

pool_size = 8
print("Processes\t%d" % pool_size)
pool = multiprocessing.Pool(processes=pool_size)

with open("apkfile_names_genome.txt", "r") as f:
    for apk_file in f.readlines():
        try:
            apk_file = apk_file.rstrip()
            lrd = literadar.LibRadarLite(apk_file)

            # lrd = literadar.LibRadarLite(apk_file)
            market_name = apk_file.split("/")[-3]
            writer_name = apk_file.split("/")[-2]
            extracted_path = "./smali" + "/%s/%s/%s" % (market_name, writer_name, lrd.HEX_SHA256)

            #apkFeature = get_appauth_features(lrd, extracted_path)
            apkFeature = pool.apply_async(get_appauth_spl_features, (lrd, extracted_path,))
            #print("apkfeature: ",apkFeature.get())
            authfeats_dict[apk_file] = apkFeature.get()

            # print("\n\n ./smali/%s/%s/%s" % (market_name, writer_name, lrd.HEX_SHA256),"\n\n")
            shutil.rmtree("./smali" + "/%s/%s/%s" % (market_name, writer_name, lrd.HEX_SHA256), ignore_errors=True)
            f_log.write("./smali" + "/%s/%s/%s" % (market_name, writer_name, lrd.HEX_SHA256) + "folder is deleted!"+"\n")

        except Exception as e:
            f_log.write("Error." + str(apk_file) + "\n" + str(e)+"\n")
            authfeats_dict[apk_file] = [np.nan] * 710
            # shutil.rmtree("./smali" + "/%s/%s/%s" % (market_name, writer_name, lrd.HEX_SHA256), ignore_errors=True)
            # print("./smali"+ "/%s/%s/%s" % (market_name, writer_name, lrd.HEX_SHA256),"folder is deleted!")

pool.close()
pool.join()

json.dump(authfeats_dict, authfeats_file)
authfeats_file.close()
f_log.close()
