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
    |> Seq.map (fun (cls,factor) -> printfn "class: %A factor: %A" cls factor; (cls,factor)) //Will print the likelihood of each class type (for debugging)
    |> Seq.maxBy (snd) // get the maximum based on the FACTOR only
    |> fst // return just the class (no factor)
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
                 | "2" -> Benign
                 | "4" -> Malignant
                 | _ -> Malignant
             )
        }
    )

classify trainingDataSet { id = 1018561; clumpT = 2; cellsizeuniform = 1; cellshapeuniform = 2; margadhesion = 1; SECS = 2; barenuclei = 1; blandchromatin = 3; normalnucleoli = 1; mitoses = 1; cls = Benign} // Run for result

