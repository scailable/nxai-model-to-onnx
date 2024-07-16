/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"

static int NUM_CLASSES_YOLO = 80;

extern "C" bool NvDsInferParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV3Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloTLT(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsInferParseObjectInfo convertBBox(float& bx, float& by, float& bw,
                                     float& bh, int& stride, uint& netW,
                                     uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposal(float bx, float by, float bw, float bh,
                     uint stride, uint& netW, uint& netH, int maxIndex,
                     float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
    float* detections, std::vector<float> &anchors,
    uint gridSizeW, uint gridSizeH, uint stride, uint numBBoxes,
    uint numOutputClasses, uint& netW,
    uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                float pw = anchors[b * 2];
                float ph = anchors[b * 2 + 1];

                int numGridCells = gridSizeH * gridSizeW;
                int bbindex = y * gridSizeW + x;
                float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                float bw
                    = pw * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)]);
                float bh
                    = ph * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)]);

                float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV3Tensor(
    float* detections, std::vector<int> &mask, std::vector<float> &anchors,
    uint gridSizeW, uint gridSizeH, uint stride, uint numBBoxes,
    uint numOutputClasses, uint& netW,
    uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                float pw = anchors[mask[b] * 2];
                float ph = anchors[mask[b] * 2 + 1];

                int numGridCells = gridSizeH * gridSizeW;
                int bbindex = y * gridSizeW + x;
                float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];

                float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}

static inline std::vector<NvDsInferLayerInfo*>
SortLayers(std::vector<NvDsInferLayerInfo> & outputLayersInfo)
{
    std::vector<NvDsInferLayerInfo*> outLayers;
    for (auto &layer : outputLayersInfo) {
        outLayers.push_back (&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
        [](NvDsInferLayerInfo* a, NvDsInferLayerInfo* b) {
            return a->inferDims.d[1] < b->inferDims.d[1];
        });
    return outLayers;
}

static bool NvDsInferParseYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    std::vector<float> &anchors,
    std::vector<std::vector<int>> &masks)
{
    uint kNUM_BBOXES = 3;

    std::vector<NvDsInferLayerInfo*> sortedLayers =
        SortLayers (outputLayersInfo);

    if (sortedLayers.size() != masks.size()) {
        std::cerr << "ERROR: yoloV3 output layer.size: " << sortedLayers.size()
                  << " does not match mask.size: " << masks.size() << std::endl;
        return false;
    }

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    for (uint idx = 0; idx < masks.size(); ++idx) {
        NvDsInferLayerInfo &layer = *sortedLayers[idx]; // 255 x Grid x Grid

        assert(layer.inferDims.numDims == 3);
        uint gridSizeH = layer.inferDims.d[1];
        uint gridSizeW = layer.inferDims.d[2];
        uint stride = DIVUP(networkInfo.width, gridSizeW);
        assert(stride == DIVUP(networkInfo.height, gridSizeH));

        std::vector<NvDsInferParseObjectInfo> outObjs =
            decodeYoloV3Tensor((float*)(layer.buffer), masks[idx], anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                       NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);
        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }


    objectList = objects;

    return true;
}

static NvDsInferParseObjectInfo convertBBoxYoloV4(float& bx1, float& by1, float& bx2,
                                     float& by2, uint& netW, uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);

    return b;
}

static void addBBoxProposalYoloV4(float bx, float by, float bw, float bh,
                     uint& netW, uint& netH, int maxIndex,
                     float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV4Tensor(
    float* boxes, float* scores,
    uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    uint& netW, uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;

    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
        {
            float prob = scores[score_location + c];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = c;
            }
        }

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += detectionParams.numClassesConfigured;
    }

    return binfo;
}


/* C-linkage to prevent name-mangling */

static bool NvDsInferParseYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;
    
    NvDsInferLayerInfo &boxes = outputLayersInfo[0]; // num_boxes x 4
    NvDsInferLayerInfo &scores = outputLayersInfo[1]; // num_boxes x num_classes
    NvDsInferLayerInfo &subbox = outputLayersInfo[2];
    //* printf("%d\n", subbox.inferDims.numDims);
    // 3 dimensional: [num_boxes, 1, 4]
    assert(boxes.inferDims.numDims == 3);
    // 2 dimensional: [num_boxes, num_classes]
    assert(scores.inferDims.numDims == 2);

    // The second dimension should be num_classes
    assert(detectionParams.numClassesConfigured == scores.inferDims.d[1]);
    
    uint num_bboxes = boxes.inferDims.d[0];

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV4Tensor(
            (float*)(boxes.buffer), (float*)(scores.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV4 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseCustomYoloV3(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static std::vector<float> kANCHORS = {
        10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
        45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    static std::vector<std::vector<int>> kMASKS = {
        {6, 7, 8},
        {3, 4, 5},
        {0, 1, 2}};
    return NvDsInferParseYoloV3 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}

extern "C" bool NvDsInferParseCustomYoloV3Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static std::vector<float> kANCHORS = {
        10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    static std::vector<std::vector<int>> kMASKS = {
        {3, 4, 5},
        //{0, 1, 2}}; // as per output result, select {1,2,3}
        {1, 2, 3}};

    return NvDsInferParseYoloV3 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kANCHORS, kMASKS);
}

static bool NvDsInferParseYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // copy anchor data from yolov2.cfg file
    std::vector<float> anchors = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
        5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
    uint kNUM_BBOXES = 5;

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    assert(layer.inferDims.numDims == 3);
    uint gridSizeH = layer.inferDims.d[1];
    uint gridSizeW = layer.inferDims.d[2];
    uint stride = DIVUP(networkInfo.width, gridSizeW);
    assert(stride == DIVUP(networkInfo.height, gridSizeH));
    for (auto& anchor : anchors) {
        anchor *= stride;
    }
    std::vector<NvDsInferParseObjectInfo> objects =
        decodeYoloV2Tensor((float*)(layer.buffer), anchors, gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                   NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseCustomYoloTLT(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int topK = 200;
    int* keepCount = static_cast <int*>(outputLayersInfo.at(0).buffer);
    float* boxes = static_cast <float*>(outputLayersInfo.at(1).buffer);
    float* scores = static_cast <float*>(outputLayersInfo.at(2).buffer);
    float* cls = static_cast <float*>(outputLayersInfo.at(3).buffer);

    for (int i = 0; (i < keepCount[0]) && (objectList.size() <= topK); ++i)
    {
        float* loc = &boxes[0] + (i * 4);
        float* conf = &scores[0] + i;
        float* cls_id = &cls[0] + i;

        if(conf[0] > 1.001)
            continue;

        if((loc[0] < 0) || (loc[1] < 0) || (loc[2] < 0) || (loc[3] < 0))
            continue;

        if((loc[0] > networkInfo.width) || (loc[2] > networkInfo.width) || (loc[1] > networkInfo.height) || (loc[3] > networkInfo.width))
           continue;

        if((loc[2] < loc[0]) || (loc[3] < loc[1]))
            continue;

        if(((loc[3] - loc[1]) > networkInfo.height) || ((loc[2]-loc[0]) > networkInfo.width))
            continue;

        NvDsInferParseObjectInfo curObj{static_cast<unsigned int>(cls_id[0]),
                                        loc[0],loc[1],(loc[2]-loc[0]),
                                        (loc[3]-loc[1]), conf[0]};
        objectList.push_back(curObj);

    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3Tiny);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloTLT);
