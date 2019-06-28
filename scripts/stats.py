#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Output stats on distributions")
    parser.add_argument("--train",
                        help="Path to ruler.csv file",
                        required=True)
    parser.add_argument("--test",
                        help="Path to test.csv file",
                        required=True)
    parser.add_argument("--cover",
                        help="Path to cover.csv file",
                        required=True)
    args=parser.parse_args()

    train=pd.read_csv(args.train)
    test=pd.read_csv(args.test)
    lenTrain=len(train)
    lenTest=len(test)
    percTrain=(float(lenTrain)/(lenTrain+lenTest))*100
    percTest=(float(lenTest)/(lenTrain+lenTest))*100
    percTrain=round(percTrain,1)
    percTest=round(percTest,1)
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle("Population Distribution (by localization)")
    train.hist(column="species_id", ax=ax1)
    test.hist(column="species_id", ax=ax2)
    ax1.set_title("Training Data ({} %)".format(percTrain))
    ax1.set_xlabel("Species ID")
    ax1.set_ylabel("Count")
    ax2.set_title("Test Data ({} %)".format(percTest))
    ax2.set_xlabel("Species ID")
    ax2.set_ylabel("Count")


    cover=pd.read_csv(args.cover)
    coverFig = plt.figure()
    covAx=plt.axes()
    coverFig.suptitle("Quality Distribution")
    cover.hist(column="cover", ax=covAx)
    covAx.set_title('')
    covAx.set_xticks([0,1,2])
    covAx.set_xticklabels(['No Fish','Hand Covering', 'Clear'])
    plt.show()
