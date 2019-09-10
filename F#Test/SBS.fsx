////Type Definitions:
//Classification types
type Class =
    | D1
    | D2
    | D3
    | D4

//Data format for sample data
type Data ={
    //id:int
    a0:int  
    a1:int  
    a2:int  
    a3:int  
    a4:int  
    a5:int  
    a6:int  
    a7:int  
    a8:int  
    a9:int  
    a10:int  
    a11:int 
    a12:int  
    a13:int  
    a14:int  
    a15:int  
    a16:int  
    a17:int 
    a18:int 
    a19:int 
    a20:int  
    a21:int 
    a22:int  
    a23:int  
    a24:int  
    a25:int  
    a26:int  
    a27:int 
    a28:int 
    a29:int 
    a30:int  
    a31:int 
    a32:int  
    a33:int  
    a34:int  
    cls:Class
    raw:string []
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
    let d = 1//35-14   //number of attributes - the number of singleton values
    (Q dataSet cls)
    *(F dataSet d (fun x -> x.a0) sample.a0 cls)
    *(F dataSet d (fun x -> x.a1) sample.a1 cls)
    *(F dataSet d (fun x -> x.a2) sample.a2 cls)
    *(F dataSet d (fun x -> x.a3) sample.a3 cls)
    *(F dataSet d (fun x -> x.a4) sample.a4 cls)
    *(F dataSet d (fun x -> x.a5) sample.a5 cls)
    *(F dataSet d (fun x -> x.a6) sample.a6 cls)
    *(F dataSet d (fun x -> x.a7) sample.a7 cls)
    *(F dataSet d (fun x -> x.a8) sample.a8 cls)
    *(F dataSet d (fun x -> x.a9) sample.a9 cls)
    // *(F dataSet d (fun x -> x.a10) sample.a10 cls)    //Not relavent
    *(F dataSet d (fun x -> x.a11) sample.a11 cls)  
    // *(F dataSet d (fun x -> x.a12) sample.a12 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a13) sample.a13 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a14) sample.a14 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a15) sample.a15 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a16) sample.a16 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a17) sample.a17 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a18) sample.a18 cls)    //Not relavent
    *(F dataSet d (fun x -> x.a19) sample.a19 cls)
    *(F dataSet d (fun x -> x.a20) sample.a20 cls)
    *(F dataSet d (fun x -> x.a21) sample.a21 cls)
    *(F dataSet d (fun x -> x.a22) sample.a22 cls)
    *(F dataSet d (fun x -> x.a23) sample.a23 cls)
    *(F dataSet d (fun x -> x.a24) sample.a24 cls)
    *(F dataSet d (fun x -> x.a25) sample.a25 cls)
    *(F dataSet d (fun x -> x.a26) sample.a26 cls)
    *(F dataSet d (fun x -> x.a27) sample.a27 cls)
    // *(F dataSet d (fun x -> x.a28) sample.a28 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a29) sample.a29 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a30) sample.a30 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a31) sample.a31 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a32) sample.a32 cls)    //Not relavent
    // *(F dataSet d (fun x -> x.a33) sample.a33 cls)    //Not relavent
    *(F dataSet d (fun x -> x.a34) sample.a34 cls)
    
    //let d = number of attributes
    //(Q dataSet cls)
    //*(F dataSet d (fun x -> x.att1) sample.att1 cls)
    //*(F dataSet d (fun x -> x.att2) sample.att2 cls)
    //*(F dataSet d (fun x -> x.att3) sample.att3 cls)
    // ...
    //*(F dataSet 1 (fun x -> x.attd) sample.attd cls)

//Actually classifies a sample datapoint into a class.
let classify (dataSet:DataSet) (sample:Data) =
    if dataSet |> Seq.isEmpty then failwithf "dataSet is EMPTY!"

    [
        Class.D1 // this should be a list of all possible classifications
        Class.D2
        Class.D3
        Class.D4
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
        |>Seq.sumBy(function              // take each element and match them with... //this is another form of the match function (equivelent to "match (x,y) with")
            | D1,D1         //If you have D1*D1, D2*D2, D3*D3, D4*D4, then 0.
            | D2,D2
            | D3,D3
            | D4,D4 -> 0.   //This cleans stuff up way better for when you have more than 2 classes
            | _ -> 1.       //If LITERALLY ANYTHING ELSE, then return 1

            )               //Then sum the results for each tuple in x
    (1.0/(float d))*(sum)           //here we're just doing the MSE calculation 1/d*SUM((Yi-'Yi)^2; i=1; i->d)
    
    //in a nutshell this gets the % of classes that were guessed incorrectly therefore... ~(0 < result < 1) //You can get get 0.0 and 1.0 but the chance is incredibly low.
    //**Note** This function *may* be wrong when doing analysis on the data we found that the error was lower than it should be for some values and we aren't sure why yet.
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
    System.IO.File.ReadAllLines(@"E:\Project 1\Data\4\soybean-small.data") // this give you back a set of line from the file (replace with your directory)
    |> Seq.map (fun line -> line.Split(',') |> Array.map (fun value -> value.Trim())) // this give you an array of elements from the comma seperated fields. We trim to make sure that any white space is removed.
    |> Seq.filter (Seq.exists(fun f -> f="?") >> not)   //This filters out all lines that contain a "?"
    |> Seq.map (fun fields ->   //This will map the lines to objects returning a seqence of datapoints (or a DataSet as defined above)
        {
            //id = fields.[0] |> System.Int32.Parse
            a0 = fields.[0] |> System.Int32.Parse
            a1 = fields.[1] |> System.Int32.Parse 
            a2 = fields.[2] |> System.Int32.Parse  
            a3 = fields.[3] |> System.Int32.Parse  
            a4 = fields.[4] |> System.Int32.Parse  
            a5 = fields.[5] |> System.Int32.Parse  
            a6 = fields.[6] |> System.Int32.Parse  
            a7 = fields.[7] |> System.Int32.Parse  
            a8 = fields.[8] |> System.Int32.Parse  
            a9 = fields.[9] |> System.Int32.Parse  
            a10 = fields.[10] |> System.Int32.Parse  
            a11 = fields.[11] |> System.Int32.Parse 
            a12 = fields.[12] |> System.Int32.Parse  
            a13 = fields.[13] |> System.Int32.Parse  
            a14 = fields.[14] |> System.Int32.Parse  
            a15 = fields.[15] |> System.Int32.Parse  
            a16 = fields.[16] |> System.Int32.Parse  
            a17 = fields.[17] |> System.Int32.Parse  
            a18 = fields.[18] |> System.Int32.Parse  
            a19 = fields.[19] |> System.Int32.Parse  
            a20 = fields.[20] |> System.Int32.Parse  
            a21 = fields.[21] |> System.Int32.Parse 
            a22 = fields.[22] |> System.Int32.Parse  
            a23 = fields.[23] |> System.Int32.Parse  
            a24 = fields.[24] |> System.Int32.Parse  
            a25 = fields.[25] |> System.Int32.Parse  
            a26 = fields.[26] |> System.Int32.Parse  
            a27 = fields.[27] |> System.Int32.Parse  
            a28 = fields.[28] |> System.Int32.Parse  
            a29 = fields.[29] |> System.Int32.Parse  
            a30 = fields.[30] |> System.Int32.Parse  
            a31 = fields.[31] |> System.Int32.Parse 
            a32 = fields.[32] |> System.Int32.Parse  
            a33 = fields.[33] |> System.Int32.Parse  
            a34 = fields.[34] |> System.Int32.Parse   
            cls = fields.[35] |> (fun x -> 
                 match x with
                 | "D1" -> D1 // if D1 then D1
                 | "D2" -> D2 // ...
                 | "D3" -> D3 // ...
                 | "D4" -> D4 // ...
                 | _ -> D1    // if it's anything else then make it a D1 (I need a default case)
             )
            raw=fields
        }
    )

//classify trainingDataSet { id = 1018561; clumpT = 2; cellsizeuniform = 1; cellshapeuniform = 2; margadhesion = 1; SECS = 2; barenuclei = 1; blandchromatin = 3; normalnucleoli = 1; mitoses = 1; cls = Benign} // Run for result
doKFold 10 trainingDataSet //does a single 10-fold cross validation

#load @"..\Tools\Clipboard.fsx"
open Clipboard

//Run the kfold test 100 times, take the average 
//Yes this takes a while timed @ 00:04:09.47 
Seq.init 45 (fun k ->   //do this for all possible 'k's
    (k+2),
    Seq.init 100 (fun _ -> doKFold (k+2) trainingDataSet)
    |>Seq.average
)
|> Seq.map (fun (k,mse) -> sprintf "%d\t%f" k mse)
|> String.concat "\n"
|> toClipboard //Sends to clipboard after you see "val it : unit()" then you can CTRL-V that sh*t anywhere (Probs Excel tho) (Please clean this up before submission)

//Including the singleton values in the F function forces the average score to lie around 0.2026 an error of  20.26% (with k = 10)
//Excluding the singleton values in the F function forces the average score to lie around 0.0275 an error of   2.75% (with k = 10)

//This, we believe is due to the +1 we add to each attribute where we assume that there *may* be a single example that fits any value
//and that this fact of the F function leads to the algorithm being more uncertain about some values than it should be

//"Collecting information that is irrelevent is worse than not collecting information at all" - This Algorithm

