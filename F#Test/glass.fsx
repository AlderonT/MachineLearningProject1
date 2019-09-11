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

    RI      : float     // 1.5112 to 1.5339
    NA2O    : float     // 10.73 to 17.38
    MGO     : float     // 0 to 4.49
    AL2O3   : float     // 0.29 to 3.5
    SIO2    : float     // 69.81 to 75.41
    K2O     : float     // 0 to 6.21
    CAO     : float     // 5.43 to 16.19
    BAO     : float     // 0 to 3.15
    FE2O3   : float     // 0 to 0.51

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
    |> Seq.mapi (fun i x -> printfn "i = %A loss: %A" i x; x)   //Just printing the % of failures for each subset (debuging code)  ////DEBUG Remove before submission
    |> Seq.average                          //the result is a seq of floats so we'll just get the average our % failuresto give us a result to our k-fold analysis as the accuracy of our algorithm

////

//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    System.IO.File.ReadAllLines(@"C:\Users\chris\MEGASync\MSU\CSCI447\assignment_1\MachineLearningProject1\Data\2\glass.data") // this give you back a set of line from the file (replace with your directory)
    |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim())) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
    |> Seq.filter (Seq.exists(fun f -> f="?") >> not)   //This filters out all lines that contain a "?"
    |> Seq.map (fun fields ->   //This will map the lines to objects returning a seqence of datapoints (or a DataSet as defined above)
        {
            ID = fields.[0] |> System.Int32.Parse
            RI = float fields.[1]                                   // Typecast floats
            NA2O = float fields.[2]
            MGO = float fields.[3] 
            AL2O3 = float fields.[4] 
            SIO2 = float fields.[5] 
            K2O = float fields.[6]
            CAO = float fields.[7]
            BAO = float fields.[8]
            FE2O3 = float fields.[9]

            CLS = fields.[10] |> (fun x -> 
                 match x with
                 | "1" -> BuildingWindowsFloatProcessed
                 | "2" -> BuildingWindowsNonFloatProcessed
                 | "3" -> VehicleWindowsFloatProcessed
                 | "4" -> VehicleWindowsNonFloatProcessed
                 | "5" -> Containers
                 | "6" -> Tableware
                 | "7" -> Headlamps
                 | _   -> VehicleWindowsFloatProcessed          // Since there are none of these in the current dataset, this can serve as a default
             )
        }
    )


//classify trainingDataSet { id = 1018561; clumpT = 2; cellsizeuniform = 1; cellshapeuniform = 2; margadhesion = 1; SECS = 2; barenuclei = 1; blandchromatin = 3; normalnucleoli = 1; mitoses = 1; cls = Benign} // Run for result
doKFold 10 trainingDataSet

//As things stand right now, executing everything you will get a number between 0. and 1.0 (though most numbers lie between 0.0 and 0.1 with an average ~0.02) //This is a good number 2% is a low fail rate
//This result gives the average % of failures for all validation sets.
