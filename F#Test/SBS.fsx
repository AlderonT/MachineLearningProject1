//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Allen Simpson; Chris Major; Tysen Radovich; Farshina Nazrul
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Soybean (small) dataset (soybean-small.data)
//--------------------------------------------------------------------------------------------------------------------------


//importing the Naive-Bayes implementation
#load @"naiveBayes.fsx"
open NaiveBayes

let trainingDataSet =
    normaltrainingDataSet (@"E:\Project 1\Data\4\soybean-small.data") true (None) 35
    
let withoutSingletons= 
    let attributeCount = trainingDataSet |> Seq.head |> (fun x -> x.attributes.Length)
    let attributesToRemove =
        Seq.init attributeCount (fun i -> 
            i,
            trainingDataSet
            |> Seq.map (fun x -> x.attributes.[i])
            |> Seq.countBy id 
            |> Seq.length
        )
        |>Seq.filter (fun (_,x) -> x=1)
        |>Seq.map fst
        |>Seq.toArray
    trainingDataSet
    |> Seq.map (fun x -> 
        let attrs = 
            x.attributes 
            |> Seq.mapi (fun i v -> i,v) 
            |> Seq.choose (function 
                | i,_ when (attributesToRemove|>Seq.contains i) -> None 
                | _,v -> Some v 
            )
            |>Seq.toArray
        {x with attributes = attrs}        
    )    

let shuffledTrainingDataSet ()= 
    shuffledtrainingDataSet (@"E:\Project 1\Data\4\soybean-small.data") true (None) 4  

let shuffledWithoutSingletons ()= 
    let trainingDataSet = shuffledTrainingDataSet()
    let attributeCount = trainingDataSet |> Seq.head |> (fun x -> x.attributes.Length)
    let attributesToRemove =
        Seq.init attributeCount (fun i -> 
            i,
            trainingDataSet
            |> Seq.map (fun x -> x.attributes.[i])
            |> Seq.countBy id 
            |> Seq.length
        )
        |>Seq.filter (fun (_,x) -> x=1)
        |>Seq.map fst
        |>Seq.toArray
    trainingDataSet
    |> Seq.map (fun x -> 
        let attrs = 
            x.attributes 
            |> Seq.mapi (fun i v -> i,v) 
            |> Seq.choose (function 
                | i,_ when (attributesToRemove|>Seq.contains i) -> None 
                | _,v -> Some v 
            )
            |>Seq.toArray
        {x with attributes = attrs}        
    )    

let sw = System.Diagnostics.Stopwatch.StartNew ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingDataSet)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

sw.Start ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  withoutSingletons)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Without Singleton attributes
//Average error: 2.6900% +/- ~0.3%
//time: 00:00:04.9346 
//With Singleton attributes
//Average error: 20.7250% +/- ~0.3%
//time: 00:00:08.0298


sw.Start ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10 (shuffledTrainingDataSet ()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed


sw.Start ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10 (shuffledWithoutSingletons ()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed
 
//Without Singleton attributes
//Average error: 36.0800% +/- ~0.4%
//time: 00:00:14.9671
//With Singleton attributes
//Average error: 32.1350% +/- ~0.3%
//time: 00:00:16.2004
