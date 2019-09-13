//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Chris Major
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Glass dataset (glass.data)
//--------------------------------------------------------------------------------------------------------------------------

//importing the Naive-Bayes implementation
#load @"naiveBayes.fsx"
open NaiveBayes


////

//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    normaltrainingDataSetForRealValuedAttributes 8 (@"E:\Project 1\Data\2\glass.data") true (Some 0) 10  

let shuffledTrainingDataSet ()= 
    shuffledtrainingDataSetForRealValuedAttributes 8 (@"E:\Project 1\Data\2\glass.data") true (Some 0) 10  



let sw = System.Diagnostics.Stopwatch.StartNew ()
Seq.init 10 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingDataSet)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 35.1340% +/- ~2%
//time: 00:01:55.5431


sw.Start ()
Seq.init 10 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  (shuffledTrainingDataSet()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed


//Average error: 36.2554% +/- ~2%
//time: 00:03:48.3773

