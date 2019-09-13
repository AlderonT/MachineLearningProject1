//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Farshina Nazrul
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository House Votes '84 dataset (house-votes-84.data)
//--------------------------------------------------------------------------------------------------------------------------


//importing the Naive-Bayes implementation
#load @"naiveBayes.fsx"
open NaiveBayes


//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    normaltrainingDataSet (@"E:\Project 1\Data\5\house-votes-84.data") false (None) 0  

let newShuffledTrainingDataSet () = 
    shuffledtrainingDataSet (@"E:\Project 1\Data\5\house-votes-84.data") false (None) 0  


//Due to time complexity we chose to run the tests on votesPerformance. 
//The results for any given test should be the exact same given the same inputs
//however votesPerformance runs in significantly faster time.

//Run analysis:

let sw = System.Diagnostics.Stopwatch.StartNew ()
Seq.init 10 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingDataSet)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 10.0666% 
//time: 00:00:37.0354 (only 10 repeats)


sw.Start ()
Seq.init 10 (fun k -> printfn "Working on %d..." (k+1); doKFold 10 (newShuffledTrainingDataSet ()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed
 
//Average error: 10.2743% +/- ~0.05%
//time: 00:1:14.7231 (only 10 repeats)
