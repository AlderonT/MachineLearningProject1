//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Allen Simpson
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Breast Cancer Wisconsin dataset (breast-cancer-wisconsin.data)
//--------------------------------------------------------------------------------------------------------------------------

//importing the Naive-Bayes implementation
#load @"naiveBayes.fsx"
open NaiveBayes



//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    normaltrainingDataSet (@"E:\Project 1\Data\1\breast-cancer-wisconsin.data") true (Some 0) 10  

let newShuffledTrainingDataSet () = 
    shuffledtrainingDataSet (@"E:\Project 1\Data\1\breast-cancer-wisconsin.data") true (Some 0) 10  

//Run analysis:

let sw = System.Diagnostics.Stopwatch.StartNew ()
Seq.init 1 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingDataSet)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 2.4604% +/- ~0.3%
//time: 00:01:14.7477 (only 10 repeats)


sw.Start ()
Seq.init 1 (fun k -> printfn "Working on %d..." (k+1); doKFold 10 (newShuffledTrainingDataSet ()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed
 
//Average error: 2.5473% +/- ~0.05%
//time: 00:10:45.9402 (only 10 repeats)