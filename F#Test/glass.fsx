//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Chris Major
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Glass dataset (glass.data)
//--------------------------------------------------------------------------------------------------------------------------

// Type Definitions:
//--------------------------------------------------------------------------------------------------------------------------

// Classification types (Class)
type Class =
    | BuildingWindowsFloatProcessed
    | BuildingWindowsNonFloatProcessed
    | VehicleWindowsFloatProcessed
    | VehicleWindowsNonFloatProcessed           // Note: none of these values in glass.data
    | Containers
    | Tableware
    | Headlamps

// Data format for sample data (Attribute)
type Data = {
    ID      : int       // 1 to 214

    RI      : int     // 1.5112 to 1.5339
    NA2O    : int     // 10.73 to 17.38
    MGO     : int     // 0 to 4.49
    AL2O3   : int     // 0.29 to 3.5
    SIO2    : int     // 69.81 to 75.41
    K2O     : int     // 0 to 6.21
    CAO     : int     // 5.43 to 16.19
    BAO     : int     // 0 to 3.15
    FE2O3   : int     // 0 to 0.51

    CLS     : Class     // Class Value
} 

// Type alias for the training set
type DataSet = Data seq

// Functions:
//--------------------------------------------------------------------------------------------------------------------------

// Implement #{pred}, the representation of the number of elements in the set where pred is true
let filteredCount pred (s:'a seq) = s |> Seq.filter pred |> Seq.length

// Implements Q (C = ci) = #{pred} / N, the percentage of elements in the data set that fall into class "cls"
let Q (dataSet:DataSet) cls = 
    (float (filteredCount (fun x -> x.CLS = cls) dataSet)) / (float (dataSet|>Seq.length))

// Implements F (Aj = ak, C = ci) = #{(xaj = ak) & (x in ci)} + 1 / N + d
// Finds the likeliness that a certain attribute "Aj" has the value ak and fall into class "cls" 
let F (dataSet:DataSet) d Aj ak cls =
    let Nc = filteredCount (fun x -> x.CLS = cls) dataSet           //gets the number of elements that fall into class "cls" 
    
    let pred (x:Data) = (Aj x = ak) && (x.CLS = cls)                // determines the predicate of the F function

    (float ((filteredCount pred dataSet)+1)) / (float (Nc + d))     // executes the function F

// Implements C(x) = Q(C=ci)*Product(F(Aj=ak,C=ci)) from j=1 to d
// Finds the likeliness that the sample data point is of the class "cls".
let C (dataSet:DataSet) (cls:Class) (sample:Data) = 
    //for more than one attribute, additional F parts will need to be added
    let d = 9   //number of attributes

    (Q dataSet cls)
    *(F dataSet d (fun x -> x.RI) sample.RI cls)
    *(F dataSet d (fun x -> x.NA2O) sample.NA2O cls)
    *(F dataSet d (fun x -> x.MGO) sample.MGO cls)
    *(F dataSet d (fun x -> x.AL2O3) sample.AL2O3 cls)
    *(F dataSet d (fun x -> x.SIO2) sample.SIO2 cls)
    *(F dataSet d (fun x -> x.K2O) sample.K2O cls)
    *(F dataSet d (fun x -> x.CAO) sample.CAO cls)
    *(F dataSet d (fun x -> x.BAO) sample.BAO cls)
    *(F dataSet d (fun x -> x.FE2O3) sample.FE2O3 cls)



// Function to classify a data point given.
let classify (dataSet:DataSet) (sample:Data) =
    [
        // List of all possible classifications
        Class.BuildingWindowsFloatProcessed
        Class.BuildingWindowsNonFloatProcessed
        Class.VehicleWindowsFloatProcessed
        Class.VehicleWindowsNonFloatProcessed
        Class.Containers
        Class.Tableware
        Class.Headlamps


    ]
    |> Seq.map (fun cls -> cls, C dataSet cls sample)   //maps the class to the likeliness
    //|> Seq.map (fun (cls,factor) -> printfn "class: %A factor: %A" cls factor; (cls,factor)) //Will print the likelihood of each class type (for debugging)
    |> Seq.maxBy (snd) // get the maximum based on the FACTOR only
    |> fst // return just the class (no factor)
////    

//// Loss Function

let MSE d (x:(Class*Class) seq) =   //This is the MSE (Mean Square Error) loss function, it takes the number of elements (|validationSet| from below), and a sequence of (class*class) tuples
    let sum =                           //We start out by getting the sums for the second part
        x                               //take our seq of class*class
        |>Seq.map(function              // take each element and match them with... //this is another form of the match function (equivelent to "match (x,y) with")
            | BuildingWindowsFloatProcessed, BuildingWindowsFloatProcessed          // correct 0 error
            | BuildingWindowsNonFloatProcessed, BuildingWindowsNonFloatProcessed    // wrong 1 error
            | VehicleWindowsFloatProcessed, VehicleWindowsFloatProcessed            // wrong -1 error (this is just how I did the math out on the side)
            | VehicleWindowsNonFloatProcessed, VehicleWindowsNonFloatProcessed
            | Containers, Containers
            | Tableware, Tableware
            | Headlamps, Headlamps -> 0. // correct 0 error
            | _ -> 1.
            )
        |> Seq.map (fun x -> x*x)       //then we square the values
        |> Seq.sum                      //and sum them all

    (1.0/(float d))*(sum)           //here we're just doing the MSE calculation 1/d*SUM((Yi-'Yi)^2; i=1; i->d)
    //in a nutshell this gets the % of classes that were guessed incorrectly therefore... ~(0 < result < 1) //You can get get 0.0 and 1.0 but the chance is incredibly low

////
 

//// k-fold

let getRandomFolds k (dataSet:'a seq) = //k is the number of slices dataset is the unsliced dataset
    let rnd = System.Random()           //init randomnumbergenerator
    let data = ResizeArray(dataSet)     //convert our dataset to a resizable array
    let getRandomElement() =            //Get a random element out of data
        if data.Count <= 0 then None    //if our data is empty return nothing
        else
            let idx = rnd.Next(0,data.Count)    //get a random index between 0 and |data|
            let e = data.[idx]                  //get the element e from idx
            data.RemoveAt(idx) |> ignore        //remove the element e from data
            Some e                              //return e
    let folds = Array.init k (fun _ -> Seq.empty)       //resultant folds array init as an empty seq
    let rec generate  j =                               //recursively generate an array that increments on j (think like a while loop)
        match getRandomElement() with                   //match the random element with:
        | None -> folds                                 //if there is nothing there then return folds
        | Some e ->                                     // if there is something there
            let s = folds.[j%k]                         // get the (j%k)th fold  in the array
            folds.[j%k] <- seq { yield! s; yield e }    //create a new seqence containing the old sequence (at j%k) and the new element e, and put it back into slot (j%k)
            generate (j+1)                              //increment j and run again
    generate 0                                          //calls the generate function

let applyKFold (trainingSet:Data seq) (validationSet: Data seq) =   //apply the loss function (MSE) to the kth fold
    validationSet                                                   //take our validation set
    |> Seq.map (fun x -> (classify trainingSet x,x.CLS))            //grab each element out of it and run it as the "sample" in our classify function and pair the resultant class with the element's ACTUAL class in a tuple
    |> MSE (validationSet |> Seq.length)                            //run the MSE algorithm with d = |validationSet| and the sequence of class tuples
    //                                                              //The result is a float: the % of elements that were guessed incorrectly

let doKFold k (dataSet:Data seq)=           //This is where we do the k-folding algorithim this will return the average from all the kfolds
    let folds = getRandomFolds k dataSet    //Lets get the k folds randomly using the function above; this returns an array of Data seqences
    Seq.init k (fun k ->                    //We're initializing a new seq of size k using the lambda function "(fun k -> ...)"" to get the kth element
        let validationSet = folds.[k]       //The first part of our function we get the validation set by grabing the kth data Seq from folds
        let trainingSet =                   //The training set requires us to do a round-about filter due to the fact that seqences are never identical and we can't use a Seq.filter...
            folds                           //lets grab the array of data seqences
            |> Seq.mapi (fun i f -> (i,f))  //each seqence in the array is mapped to a tuple with the index of the sequence as "(index,sequence)"
            |> Seq.filter(fun (i,_) -> i<>k)//now we will filter out the seqence that has the index of k
            |> Seq.collect snd              //now we grab the seqence from the tuple
        applyKFold trainingSet validationSet//Finally lets apply our function above "applyKFold" to our training set and validation set
    )
    //|> Seq.mapi (fun i x -> printfn "i = %A loss: %A" i x; x)   //Just printing the % of failures for each subset (debuging code)  ////DEBUG Remove before submission
    |> Seq.average                          //the result is a seq of floats so we'll just get the average our % failuresto give us a result to our k-fold analysis as the accuracy of our algorithm

////

let newTrainingDataSet n =                                                                          //This is a variation on the trainingDataSet that we made to accomidate for continuous values for the attributes 
    let data =
        System.IO.File.ReadAllLines(@"E:\Project 1\Data\2\glass.data")                              // this give you back a set of line from the file (replace with your directory)
        |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim()))           // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
        |> Seq.filter (Seq.exists(fun f -> f="?") >> not)                                           //This filters out all lines that contain a "?"
        |> Seq.map (fun sa ->                                                                       //here we are taking each array of strings (the attributes)
            let i = sa.[0] |> System.Int32.Parse                                                    //parse the first element (the id) as an int
            let cls = sa.[sa.Length-1] |> System.Int32.Parse                                        //parse the last element (the class) as an int
            let attrs = sa|> Seq.skip 1 |>Seq.take 9 |> Seq.map System.Double.Parse |> Seq.toArray  //take the array, skip the first and take all but the last, parse each value as a double and make the result into an array
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
            ID = i                                              //ID is i 

            RI = xs.[0] |> funcs.[0]                            //for each attribute in the attribute array,
            NA2O = xs.[1] |> funcs.[1]                          //get the attribute's index, access that attribute value
            MGO = xs.[2] |> funcs.[2]                           //and pass it to the attribute conversion function (funcs)
            AL2O3 = xs.[3] |> funcs.[3]                         //this takes each individual attribute float and "classifies" it
            SIO2 = xs.[4] |> funcs.[4]
            K2O = xs.[5] |> funcs.[5]
            CAO = xs.[6] |> funcs.[6]
            BAO = xs.[7] |> funcs.[7]
            FE2O3 = xs.[8] |> funcs.[8]

            CLS = cls |> (fun x ->                              //Here we're matching cls with what class it represents
                 match x with                                   //this could be done on line 207 (when defining cls) but
                 | 1 -> BuildingWindowsFloatProcessed           //this format matches the same format we use in the other 
                 | 2 -> BuildingWindowsNonFloatProcessed        //files
                 | 3 -> VehicleWindowsFloatProcessed
                 | 4 -> VehicleWindowsNonFloatProcessed
                 | 5 -> Containers
                 | 6 -> Tableware
                 | 7 -> Headlamps
                 | _   -> VehicleWindowsFloatProcessed          // Since there are none of these in the current dataset, this can serve as a default
             )
        }
    )



let newShuffledTrainingDataSet n =                                                                  //This is a variation on the trainingDataSet that we made to accomidate for continuous values for the attributes 

    let shuffleAttributes () =                                                                  //This will generate a verson of the data that shuffles 10% of the attributes
        let workingData =                                                                       //We're getting the data we will work with
            System.IO.File.ReadAllLines(@"E:\Project 1\Data\2\glass.data")                      //get the data from file (yes this needs to match a directory that can read it)
            |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim()))   //split the lines on the commas
            |> Seq.map (fun sa ->                                                               //now we are taking each value and...
                let i = sa.[0]                                                                  //the first value gets to be an ID
                let cls = sa.[sa.Length-1]                                                      //the last value gets to be a CLS
                let attribs = sa |> Seq.skip 1 |> Seq.take (sa.Length-2) |> Seq.toArray         //We take everything else, drop the first and last values and make the result into an array
                (i,cls),attribs                                                                 //we are making a tuple of a tuple here 
            )
            |>Seq.toArray                                                                       //making the sequence into an array so we don't recalculate every time we call workingData

        let shuffle (data:string [] []) attr=                                       //this will shuffle the attribute "attr" in the string array data
            let mutable i = 0                                                       //we are doing this imperitvely as a show of force (this is how you make a mutable value)
            let attributes = ResizeArray (data |> Seq.map (fun xs -> xs.[attr]))    //we are making an array of the values from data's 'attr'th attribute array
            let rnd = System.Random()                                               //make our randomNumberGenerator
            while attributes.Count>0 do                                             //while we have attributes...
                let j = rnd.Next(0,attributes.Count)                                //get a random index in attributes
                let v = attributes.[j]                                              //assign v the value of the j'th attribute out of attributes
                attributes.RemoveAt(j)                                              //remove the j'th element from attributes
                data.[i].[attr] <- v                                                //replace the value of the i'th data point's 'attr'th attribute (this effects the actual value of data outside the function)
                i <- i+1                                                            //increment i

        
        let data = workingData |>Array.map snd                                                  //defining data as the attribute array from working Data
        let modifyCount = (workingData.[0]|> snd |> Array.length |> float)*0.1 |> ceil |> int   //this is the count of modifiable attributes (literally the length of attributes*0.1 rounded up)
        let attribsCount = (workingData.[0]|> snd |> Array.length)                              //this is the number of actual attributes
        let rnd = System.Random()                                                               //make a randomNumberGenerator
        let idxs = ResizeArray([0..(attribsCount-1)])                                           //we are making a mutable list of indicies 
        List.init modifyCount (fun _ ->                                                         //make a new list with magnitude of modify count (the number of elements we are shuffling)
            let j = rnd.Next(0,idxs.Count)                                                      //get a random index from idxs
            let i = idxs.[j]                                                                    //let i be the random index
            idxs.RemoveAt(j)                                                                    //we shall remove said index from idxs (so we don't choose it again)
            i                                                                                   //and add it to our list
        )                                                                                       ////This randomly chooses the attribute numbers we're going to shuffle
        |>Seq.iter (shuffle data)                                                               //we now iter through this list of indecies and shuffle the data at the index (This shuffling modifies the actual values of data)
        Seq.zip workingData data                                                                //then we make a tuple of the working data and the shuffled data
        |>Seq.map (fun (((i, cls),oldData),newData) ->                                          //Then we take the form ((string,string),string[],string[])
            seq {yield i; yield! newData; yield cls} |> String.concat "," )                     //and convert it into one long sequence of strings which we immediately concat with ','

    
    let data =
        shuffleAttributes ()                                                                        //we start with the shuffled values this time
        |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim()))           // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
        |> Seq.filter (Seq.exists(fun f -> f="?") >> not)                                           //This filters out all lines that contain a "?"
        |> Seq.map (fun sa ->                                                                       //here we are taking each array of strings (the attributes)
            let i = sa.[0] |> System.Int32.Parse                                                    //parse the first element (the id) as an int
            let cls = sa.[sa.Length-1] |> System.Int32.Parse                                        //parse the last element (the class) as an int
            let attrs = sa|> Seq.skip 1 |>Seq.take 9 |> Seq.map System.Double.Parse |> Seq.toArray  //take the array, skip the first and take all but the last, parse each value as a double and make the result into an array
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
            ID = i                                              //ID is i 

            RI = xs.[0] |> funcs.[0]                            //for each attribute in the attribute array,
            NA2O = xs.[1] |> funcs.[1]                          //get the attribute's index, access that attribute value
            MGO = xs.[2] |> funcs.[2]                           //and pass it to the attribute conversion function (funcs)
            AL2O3 = xs.[3] |> funcs.[3]                         //this takes each individual attribute float and "classifies" it
            SIO2 = xs.[4] |> funcs.[4]
            K2O = xs.[5] |> funcs.[5]
            CAO = xs.[6] |> funcs.[6]
            BAO = xs.[7] |> funcs.[7]
            FE2O3 = xs.[8] |> funcs.[8]

            CLS = cls |> (fun x ->                              //Here we're matching cls with what class it represents
                 match x with                                   //this could be done on line 207 (when defining cls) but
                 | 1 -> BuildingWindowsFloatProcessed           //this format matches the same format we use in the other 
                 | 2 -> BuildingWindowsNonFloatProcessed        //files
                 | 3 -> VehicleWindowsFloatProcessed
                 | 4 -> VehicleWindowsNonFloatProcessed
                 | 5 -> Containers
                 | 6 -> Tableware
                 | 7 -> Headlamps
                 | _   -> VehicleWindowsFloatProcessed          // Since there are none of these in the current dataset, this can serve as a default
             )
        }
    )


//classify trainingDataSet { id = 1018561; clumpT = 2; cellsizeuniform = 1; cellshapeuniform = 2; margadhesion = 1; SECS = 2; barenuclei = 1; blandchromatin = 3; normalnucleoli = 1; mitoses = 1; cls = Benign} // Run for result
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

//As things stand right now, executing everything you will get a number between 0. and 1.0 (though most numbers lie between 0.0 and 0.1 with an average ~0.02) //This is a good number 2% is a low fail rate
//This result gives the average % of failures for all validation sets.
