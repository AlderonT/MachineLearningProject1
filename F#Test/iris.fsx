//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Tysen Radovich
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Iris dataset (iris.data)
//--------------------------------------------------------------------------------------------------------------------------

//importing the Naive-Bayes implementation
#load @"naiveBayes.fsx"
open NaiveBayes


//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    normaltrainingDataSet (@"E:\Project 1\Data\3\iris.data") true (None) 4  

let shuffledTrainingDataSet ()= 
    shuffledtrainingDataSet (@"E:\Project 1\Data\3\iris.data") true (None) 4  


let sw = System.Diagnostics.Stopwatch.StartNew ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingDataSet)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 7.2667% 
//time: 00:00:10.2944 


sw.Start ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  (shuffledTrainingDataSet()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 10.9200%
//time: 00:00:21.81712