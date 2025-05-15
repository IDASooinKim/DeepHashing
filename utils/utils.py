import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.spatial.distance import cdist


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    valid_queries = 0

    for iter in range(num_query):
        # ground truth: 공통 라벨이 있는 retrieval 이미지
        gnd = (np.dot(queryL[iter], retrievalL.transpose()) > 0).astype(np.float32)

        # 해밍 거리 계산 및 정렬
        hamm = CalcHammingDist(qB[iter], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        # 상위 top-k 정답만 사용
        tgnd = gnd[:topk]
        if np.sum(tgnd) == 0:
            continue

        valid_queries += 1
        pos_idx = np.where(tgnd == 1)[0]

        # 정확한 AP 계산
        prec = [(i + 1) / (pos_idx[i] + 1) for i in range(len(pos_idx))]
        ap = np.mean(prec)
        topkmap += ap

    if valid_queries == 0:
        return 0.0
    return topkmap / valid_queries

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH
