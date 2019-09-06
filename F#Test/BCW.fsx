////Type Definitions:
//Classification types
type Class =
    | Benign
    | Malignant

//Data format for sample data
type Data ={
    id:int
    clumpT:int // 1 - 10
    cellsizeuniform:int // 1 - 10
    cellshapeuniform:int // 1 - 10
    margadhesion:int // 1 - 10
    SECS:int // 1 - 10
    barenuclei:int // 1 - 10
    blandchromatin:int // 1 - 10
    normalnucleoli:int // 1 - 10
    mitoses:int // 1 - 10
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
    (Q dataSet cls)
    *(F dataSet 1 (fun x -> x.clumpT) sample.clumpT cls)
    *(F dataSet 1 (fun x -> x.cellsizeuniform) sample.cellsizeuniform cls)
    *(F dataSet 1 (fun x -> x.cellshapeuniform) sample.cellshapeuniform cls)
    *(F dataSet 1 (fun x -> x.margadhesion) sample.margadhesion cls)
    *(F dataSet 1 (fun x -> x.SECS) sample.SECS cls)
    *(F dataSet 1 (fun x -> x.barenuclei) sample.barenuclei cls)
    *(F dataSet 1 (fun x -> x.blandchromatin) sample.blandchromatin cls)
    *(F dataSet 1 (fun x -> x.normalnucleoli) sample.normalnucleoli cls)
    *(F dataSet 1 (fun x -> x.mitoses) sample.mitoses cls)
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
        Class.Benign // this should be a list of all possible classifications
        Class.Malignant
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
            | Benign,Benign -> 0.       // correct 0 error
            | Malignant,Benign -> 1.    // wrong 1 error
            | Benign,Malignant -> -1.   // wrong -1 error (this is just how I did the math out on the side)
            | Malignant,Malignant -> 0. // correct 0 error
            )
        |> Seq.map (fun x -> x*x)       //then we square the values
        |> Seq.sum                      //and sum them all
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
    |> Seq.mapi (fun i x -> printfn "i = %A loss: %A" i x; x)   //Just printing the % of failures for each subset (debuging code)  ////DEBUG Remove before submission
    |> Seq.average                          //the result is a seq of floats so we'll just get the average our % failuresto give us a result to our k-fold analysis as the accuracy of our algorithm

////

//Reads data and assigns to trainingDataSet:
let trainingDataSet =
    System.IO.File.ReadAllLines(@"E:\Project 1\Data\1\breast-cancer-wisconsin.data") // this give you back a set of line from the file (replace with your directory)
    |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim())) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
    |> Seq.filter (Seq.exists(fun f -> f="?") >> not)   //This filters out all lines that contain a "?"
    |> Seq.map (fun fields ->   //This will map the lines to objects returning a seqence of datapoints (or a DataSet as defined above)
        {
            id = fields.[0] |> System.Int32.Parse
            clumpT = fields.[1] |> System.Int32.Parse
            cellsizeuniform = fields.[2] |> System.Int32.Parse// 1 - 10
            cellshapeuniform = fields.[3] |> System.Int32.Parse // 1 - 10
            margadhesion = fields.[4] |> System.Int32.Parse // 1 - 10
            SECS = fields.[5] |> System.Int32.Parse // 1 - 10
            barenuclei = fields.[6] |> System.Int32.Parse // 1 - 10
            blandchromatin = fields.[7] |> System.Int32.Parse // 1 - 10
            normalnucleoli = fields.[8] |> System.Int32.Parse // 1 - 10
            mitoses = fields.[9] |> System.Int32.Parse // 1 - 10
            cls = fields.[10] |> (fun x -> 
                 match x with
                 | "2" -> Benign //if the string is a 2 it's benign
                 | "4" -> Malignant // if it's a 4 it's malignant
                 | _ -> Malignant // if it's anything else it's being malignant to me (I need a default case)
             )
        }
    )

//classify trainingDataSet { id = 1018561; clumpT = 2; cellsizeuniform = 1; cellshapeuniform = 2; margadhesion = 1; SECS = 2; barenuclei = 1; blandchromatin = 3; normalnucleoli = 1; mitoses = 1; cls = Benign} // Run for result
doKFold 10 trainingDataSet

//As things stand right now, executing everything you will get a number between 0. and 1.0 (though most numbers lie between 0.0 and 0.1 with an average ~0.02) //This is a good number 2% is a low fail rate
//This result gives the average % of failures for all validation sets. 