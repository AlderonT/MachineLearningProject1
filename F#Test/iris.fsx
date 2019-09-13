//--------------------------------------------------------------------------------------------------------------------------
// CSCI447 - Fall 2019
// Assignment #1
// Tysen Radovich
// 
// Program to perform the Naive-Bayes algorithm on the UCI Machine Learning Repository Iris dataset (iris.data)
//--------------------------------------------------------------------------------------------------------------------------

////Type Definitions:
//Classification types
type Class =
    | Setosa
    | Versicolour
    | Virginica
///there seems to be a correlation between petal width and petal length
//Data format for sample data
type Data ={ // all W in data refer to width and all L in data refer to length 
    sepalL:float// 1.0 - 10.0
    sepalW:float // 1.0 - 10.0
    petalL:float // 1.0 - 10.0
    petalW:float // 1.0 - 10.0
    cls:Class //2 or 4
} 

//type alias for the training set
type DataSet = Data seq
////

////Functions:
//#{pred} = the count of elements in the set that pred is true
//Implements #{pred}
let filteredCount pred (s:'a seq) = s |> Seq.filter pred |> Seq.length

//Implements Q (C=ci) = #{pred}/N // Finds the percentage of elements in the data set that fall into class "cls"
let Q (dataSet:DataSet) cls = 
    (float (filteredCount (fun x -> x.cls = cls) dataSet))/(float (dataSet|>Seq.length))

//Implements F (Aj=ak,C=ci) = #{(xaj=ak)&(x in ci)}+1/N+d
//Finds the likeliness that a certain attribute "Aj" has the value ak and fall into class "cls" 
let F (dataSet:DataSet) d Aj ak cls =
    let Nc = filteredCount (fun x -> x.cls = cls) dataSet //gets the number of elements that fall into class "cls" 
    
    let pred (x:Data) = (Aj x = ak) && (x.cls = cls) // determines the predicate of the F function

    (float ((filteredCount pred dataSet)+1)) / (float (Nc + d)) // executes the function F

//Implements C(x) = Q(C=ci)*Product(F(Aj=ak,C=ci)) from j=1 to d
// Finds the likeliness that the sample data point is of the class "cls".
let C (dataSet:DataSet) (cls:Class) (sample:Data) = 
    //for more than one attribute, additional F parts will need to be added
    //sepalL:float// 1.0 - 10.0
    //sepalW:float // 1.0 - 10.0
    //petalL:float // 1.0 - 10.0
    //petalW:float // 1.0 - 10.0
    
    let d = 4   //number of attributes
    (Q dataSet cls)
    *(F dataSet d (fun x -> x.sepalL) sample.sepalL cls)
    *(F dataSet d (fun x -> x.sepalW) sample.sepalW cls)
    *(F dataSet d (fun x -> x.petalL) sample.petalL cls)
    *(F dataSet d (fun x -> x.petalW) sample.petalW cls)
    
    //let d = number of attributes
    //(Q dataSet cls)
    //*(F dataSet d (fun x -> x.att1) sample.att1 cls)
    //*(F dataSet d (fun x -> x.att2) sample.att2 cls)
    //*(F dataSet d (fun x -> x.att3) sample.att3 cls)
    // ...
    //*(F dataSet 1 (fun x -> x.attd) sample.attd cls)

//Actually classifies a sample datapoint into a class.
let classify (dataSet:DataSet) (sample:Data) =
    [
        Class.Setosa // this should be a list of all possible classifications
        Class.Versicolour
        Class.Virginica

    ]
    |> Seq.map (fun cls -> cls, C dataSet cls sample)   //maps the class to the likeliness
    //|> Seq.map (fun (cls,factor) -> printfn "class: %A factor: %A" cls factor; (cls,factor)) //Will print the likelihood of each class type (for debugging)
    |> Seq.maxBy (snd) // get the maximum based on the FACTOR only
    |> fst // return just the class (no factor)
////    

//// Loss Function

let MSE d (x:(Class*Class) seq) =   //This is the MSE (Mean Square Error) loss function, it takes the number of elements (|validationSet| from below), and a sequence of (class*class) tuples
    let sum =   
   // | Setosa
    //| Versicolour
   // | Virginica                        //We start out by getting the sums for the second part
        x                               //take our seq of class*class
        |>Seq.sumBy(function   
                   // take each element and match them with... //this is another form of the match function (equivelent to "match (x,y) with")
            | Setosa,Versicolour -> 1.       // correct 0 error
            | Setosa,Virginica  -> 1.    // wrong 1 error
            | Setosa,Setosa -> 0.
            | Versicolour,Virginica-> 1.
            | Versicolour,Setosa -> 1.
            | Versicolour,Versicolour -> 0. // wrong -1 error (this is just how I did the math out on the side)
            | Virginica,Setosa-> 1.
            | Virginica,Versicolour -> 1.
            | Virginica,Virginica -> 0.    // correct 0 error
            )
        //Can also be done as so:
        //let sum2 = 
        //    x
        //    |>Seq.sumBy (function           // we can sum the tuples by converting the tuples to floats/ints using the "match (x,y) with" or "function" mapping
        //        | Benign,Benign -> 0.       // correct 0 error
        //        | Malignant,Benign -> 1.    // wrong 1 error
        //        | Benign,Malignant -> 1.    // wrong 1 error 
        //        | Malignant,Malignant -> 0.)// correct 0 error

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
    |> Seq.map (fun x -> (classify trainingSet x,x.cls))            //grab each element out of it and run it as the "sample" in our classify function and pair the resultant class with the element's ACTUAL class in a tuple
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

//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    System.IO.File.ReadAllLines(@"E:\Project 1\Data\3\iris.data") // this give you back a set of line from the file (replace with your directory)
    |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim())) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
    |> Seq.filter (Seq.exists(fun f -> f="?") >> not)   //This filters out all lines that contain a "?"
    |> Seq.map (fun fields ->   //This will map the lines to objects returning a seqence of datapoints (or a DataSet as defined above)
        {
            sepalL = float fields.[0]
            sepalW = float fields.[1] 
            petalL = float fields.[2] // 1 - 10
            petalW = float fields.[3] // 1 - 10
            cls = fields.[4] |> (fun x -> 
                match x with
                | "Iris-setosa" -> Setosa //if the string is a type of iris
                | "Iris-versicolour" -> Versicolour // if it's a 4 it's malignant
                | "Iris-virginica"-> Virginica  // if it's anything else it's being malignant to me (I need a default case)
                | _ -> Setosa    
            )

        }
    )


let newShuffledTrainingDataSet () = 
    let shuffleAttributes () =                                                                  //This will generate a verson of the data that shuffles 10% of the attributes
        let workingData =                                                                       //We're getting the data we will work with
            System.IO.File.ReadAllLines(@"E:\Project 1\Data\3\iris.data")                      //get the data from file (yes this needs to match a directory that can read it)
            |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim()))   //split the lines on the commas
            |> Seq.map (fun sa ->                                                               //now we are taking each value and...
                let cls = sa.[sa.Length-1]                                                      //the last value gets to be a CLS
                let attribs = sa |> Seq.take (sa.Length-1) |> Seq.toArray         //We take everything else, drop the first and last values and make the result into an array
                cls,attribs                                                                 //we are making a tuple of a tuple here 
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
        |>Seq.map (fun ((cls,oldData),newData) ->                                          //Then we take the form ((string,string),string[],string[])
            seq {yield! newData; yield cls} |> String.concat "," )                     //and convert it into one long sequence of strings which we immediately concat with ','
    
    shuffleAttributes ()                                                                        //we start with the shuffled values this time
    |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim())) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
    |> Seq.filter (Seq.exists(fun f -> f="?") >> not)   //This filters out all lines that contain a "?"
    |> Seq.map (fun fields ->   //This will map the lines to objects returning a seqence of datapoints (or a DataSet as defined above)
        {
            sepalL = float fields.[0]
            sepalW = float fields.[1] 
            petalL = float fields.[2] // 1 - 10
            petalW = float fields.[3] // 1 - 10
            cls = fields.[4] |> (fun x -> 
                match x with
                | "Iris-setosa" -> Setosa 
                | "Iris-versicolour" -> Versicolour 
                | "Iris-virginica"-> Virginica  
                | _ -> Setosa    // (default case)
            )

        }
    )


let sw = System.Diagnostics.Stopwatch.StartNew ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10  trainingDataSet)
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed

//Average error: 8.3533% 
//time: 00:00:10.2944 


sw.Start ()
Seq.init 100 (fun k -> printfn "Working on %d..." (k+1); doKFold 10 (newShuffledTrainingDataSet ()))
|>Seq.average
|>printfn "Average Loss: %f"
sw.Stop()
printfn "%A" sw.Elapsed
 
//Average error: 12.3067%
//time: 00:01:22.7216