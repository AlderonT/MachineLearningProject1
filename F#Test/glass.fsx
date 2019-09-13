//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Chris Major
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Glass dataset (glass.data)
//--------------------------------------------------------------------------------------------------------------------------

#load @"naiveBayes.fsx"
open NaiveBayes

////

let newTrainingDataSet n =                                                                          //This is a variation on the trainingDataSet that we made to accomidate for continuous values for the attributes 
    let data =
        System.IO.File.ReadAllLines(@"E:\Project 1\Data\2\glass.data")                              // this give you back a set of line from the file (replace with your directory)
        |> normalizeData (Some 0) 10 
        |> Seq.filter (snd >> Seq.exists(fun f -> f="?") >> not)                                           //This filters out all lines that contain a "?"
        |> Seq.map (fun ((i,cls),attrs) ->                                                                       //here we are taking each array of strings (the attributes)
            let attrs = attrs|> Seq.skip 1 |>Seq.take 9 |> Seq.map System.Double.Parse |> Seq.toArray  //take the array, skip the first and take all but the last, parse each value as a double and make the result into an array
            i,cls,attrs                                                                             //Then make a 3-tuple of the id,class,and attribute array
        )
    let splitIntoDivisions n xs =       //This creates a classifier function that takes an attribute seq and splits it into n divisions | result is a float -> int lambda
        let min = xs |> Seq.min         //get the minimum attribute value
        let max = xs |> Seq.max         //get the maximum attribute value
        let width = (max-min)/(float n) //get the width of each class
        fun v -> int ((v-min)/width)    //return a function that takes a float v and returns the division it lies in

    let p = 
        data                            //take the data (idx,cls,attribs[])
        |> Seq.map (fun (_,_,xs)->xs)   //get the attribs
        |> Seq.head                     //get the first set of attribs
        |> Seq.length                   //and find out how many there are (they should all be the same so p is a constant)

    let funcs =                                 //here we are making an array of functions for each attribute that will divide said attribute into n divisions
        Array.init p (fun i ->                  //so first make an array of size p (the number of attributes)
            data                                //take our data...
            |> Seq.map (fun (_,_,xs)-> xs.[i])  //and extract out the ith attribute from all our datapoints into a single sequence
            |> splitIntoDivisions n             //then have splitIntoDivisions generage a function for the ith attribute set of values 
        )

    data                                                        //Now take our data (idx,cls,attribs[])
    |> Seq.map (fun (i,cls,xs) ->                               //create our Data class by doing the following:
        {
            uid = i                                              //ID is i 
            cls = cls
            attributes = xs|>Array.mapi (fun i x -> funcs.[i] x |> string)
        }
    )





let newShuffledTrainingDataSet n =                                                                  //This is a variation on the trainingDataSet that we made to accomidate for continuous values for the attributes 
    let data =
        System.IO.File.ReadAllLines(@"E:\Project 1\Data\2\glass.data")                              // this give you back a set of line from the file (replace with your directory)
        |> shuffleAttributes (Some 0) 10 
        |> Seq.filter (snd >> Seq.exists(fun f -> f="?") >> not)                                           //This filters out all lines that contain a "?"
        |> Seq.map (fun ((i,cls),attrs) ->                                                                       //here we are taking each array of strings (the attributes)
            let attrs = attrs|> Seq.skip 1 |>Seq.take 9 |> Seq.map System.Double.Parse |> Seq.toArray  //take the array, skip the first and take all but the last, parse each value as a double and make the result into an array
            i,cls,attrs                                                                             //Then make a 3-tuple of the id,class,and attribute array
        )
    let splitIntoDivisions n xs =       //This creates a classifier function that takes an attribute seq and splits it into n divisions | result is a float -> int lambda
        let min = xs |> Seq.min         //get the minimum attribute value
        let max = xs |> Seq.max         //get the maximum attribute value
        let width = (max-min)/(float n) //get the width of each class
        fun v -> int ((v-min)/width)    //return a function that takes a float v and returns the division it lies in

    let p = 
        data                            //take the data (idx,cls,attribs[])
        |> Seq.map (fun (_,_,xs)->xs)   //get the attribs
        |> Seq.head                     //get the first set of attribs
        |> Seq.length                   //and find out how many there are (they should all be the same so p is a constant)

    let funcs =                                 //here we are making an array of functions for each attribute that will divide said attribute into n divisions
        Array.init p (fun i ->                  //so first make an array of size p (the number of attributes)
            data                                //take our data...
            |> Seq.map (fun (_,_,xs)-> xs.[i])  //and extract out the ith attribute from all our datapoints into a single sequence
            |> splitIntoDivisions n             //then have splitIntoDivisions generage a function for the ith attribute set of values 
        )

    data                                                        //Now take our data (idx,cls,attribs[])
    |> Seq.map (fun (i,cls,xs) ->                               //create our Data class by doing the following:
        {
            uid = i                                              //ID is i 
            cls = cls
            attributes = xs|>Array.mapi (fun i x -> funcs.[i] x |> string)
        }
    )



let sw = System.Diagnostics.Stopwatch.StartNew ()
// Seq.init 10 id 
// |>Seq.iter (fun n ->
let n=8 //We found that using 8 divisions leads to the lowest error without sacrificing too much time                                 
let trainingSet = newTrainingDataSet (n)
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingSet)
|>Seq.average
|>printfn "Division: %d Average Loss: %f" (n+1)

sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 35.1340% +/- ~2%
//time: 00:01:55.5431


sw.Start ()
// Seq.init 10 id 
// |>Seq.iter (fun n ->
let trainingSet2 = newShuffledTrainingDataSet (n)
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingSet2)
|>Seq.average
|>printfn "Division: %d Average Loss: %f" (n+1)
sw.Stop()
printfn "%A" sw.Elapsed


//Average error: 45.7632% +/- ~2%
//time: 00:03:48.3773

