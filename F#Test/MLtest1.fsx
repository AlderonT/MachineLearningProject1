//Attribute 1 
type Color =
    | White
    | Red
    | Purple
//Classification types
type Class =
    | Pretty
    | Ugly
//Data format for sample data
type Data ={
    id:int
    // clumpT:int // 1 - 10
    // cellsizeuniform:int
    // cellshapeuniform:int
    // margadhesion:int
    // SECS:int
    // barenuclei:int
    // blandchromatin:int
    // normalnucleoli:int
    // mitoses:int
    color: Color
    ``class``:Class
} 
//type alias for the training set
type DataSet = Data seq
//training set
let dataSet =
    [
        { id = 1; color = White; ``class`` = Ugly }
        { id = 1; color = Red; ``class`` = Ugly }
        { id = 1; color = Purple; ``class`` = Pretty }
        { id = 1; color = Purple; ``class`` = Pretty }
        { id = 1; color = Red; ``class`` = Ugly }
        { id = 1; color = Red; ``class`` = Pretty }
        { id = 1; color = White; ``class`` = Ugly }
    ] 
////Worked out examples:

// Q(Pretty) = 3.0/7.0 => 0.429
// Q(Ugly) = 4.0/7.0 => 0.571

// F(White,Pretty) = (0.0+1.0)/(3.0+1.0) = 0.25
// F(White,Ugly) = (2.0+1.0)/(4.0+1.0) =   0.60
// F(Red,Pretty) = (1.0+1.0)/(3.0+1.0) =   0.50
// F(Red,Ugly) = (2.0+1.0)/(4.0+1.0) =     0.60
// F(Purple,Pretty) = (2.0+1.0)/(3.0+1.0) =0.75
// F(Purple,Ugly) = (0.0+1.0)/(4.0+1.0) =  0.20
////

//#{pred} = the count of elements in the set that pred is true
//Implements #{pred}
let filteredCount pred (s:'a seq) = s |> Seq.filter pred |> Seq.length

//Implements Q (C=ci) = #{pred}/N // Finds the percentage of elements in the data set that fall into class "cls"
let Q (dataSet:DataSet) cls = 
    (float (filteredCount (fun x -> x.``class`` = cls) dataSet))/(float (dataSet|>Seq.length))

//Implements F (Aj=ak,C=ci) = #{(xaj=ak)&(x in ci)}+1/N+d
//Finds the likeliness that a certain attribute "Aj" has the value ak and fall into class "cls" 
let F (dataSet:DataSet) d Aj ak cls =
    let Nc = filteredCount (fun x -> x.``class`` = cls) dataSet //gets the number of elements that fall into class "cls" 
    
    let pred (x:Data) = (Aj x = ak) && (x.``class`` = cls) // determines the predicate of the F function

    (float ((filteredCount pred dataSet)+1)) / (float (Nc + d)) // executes the function F

//F dataSet 1 (fun x -> x.color) Purple Pretty

//Implements C(x) = Q(C=ci)*Product(F(Aj=ak,C=ci)) from j=1 to d
// Finds the likeliness that the sample data point is of the class "cls".
let C (dataSet:DataSet) (cls:Class) (sample:Data) = 
    //for more than one attribute, additional F parts will need to be added
    (Q dataSet cls)*(F dataSet 1 (fun x -> x.color) sample.color cls)
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
        Class.Pretty // this should be a list of all possible classifications
        Class.Ugly
    ]
    |> Seq.map (fun cls -> cls, C dataSet cls sample)   //maps the class to the likeliness
    |> Seq.map (fun (cls,factor) -> printfn "class: %A factor: %A" cls factor; (cls,factor)) //Will print the likelihood of each class type (for debugging)
    |> Seq.maxBy (snd) // get the maximum based on the FACTOR only
    |> fst // return just the class (no factor)
    
classify dataSet { id = 0; color = Purple; ``class`` = Class.Ugly} // Run for result
let rnd = System.Random()
rnd.Next()