
[<AutoOpen>]
module Clipboard =
    open System.Runtime.InteropServices

    module private Native =
        [<DllImport("user32",SetLastError=true)>]
        extern bool OpenClipboard(nativeint hWnd)
        [<DllImport("user32",SetLastError=true)>]
        extern bool CloseClipboard()
        [<DllImport("user32",SetLastError=true)>]
        extern bool EmptyClipboard()
        [<DllImport("user32",SetLastError=true)>]
        extern nativeint GetClipboardData(uint32 uFormat)
        [<DllImport("user32",SetLastError=true)>]
        extern nativeint SetClipboardData(uint32 uFormat, nativeint hHem)
        [<DllImport("kernel32",SetLastError=true)>]
        extern nativeint GlobalLock(nativeint hMem)
        [<DllImport("kernel32",SetLastError=true)>]
        extern bool GlobalUnlock(nativeint hMem)
        [<DllImport("user32",SetLastError=true)>]
        extern int EnumClipboardFormats(int format)

    open Native

    let enumerateClipboardFormats() =
        let rec loop fmt =
            seq {
                let nextFmt = EnumClipboardFormats(fmt)
                //printfn "nextFmt: %d" nextFmt
                let lastErr = System.Runtime.InteropServices.Marshal.GetLastWin32Error()
                //printfn "lastErr: %d(0x%x)" lastErr lastErr
                if lastErr <> 0 then
                    if nextFmt <> 0 then
                        yield nextFmt
                        yield! loop nextFmt
                else
                    failwithf "Error calling EnumClipboardFormats: %d" lastErr
            }
        try
            if (OpenClipboard(0n)) then
                loop 0 |> Seq.toList     
            else
                [] 
        finally
            CloseClipboard() |> ignore
    let fromClipboard() =
        try
            if (OpenClipboard(0n)) then
                let hData = GetClipboardData(13u(*CF_UNICODETEXT*))
                let ptr = GlobalLock(hData)
                if (ptr <> 0n) then
                    try
                        System.Runtime.InteropServices.Marshal.PtrToStringUni(ptr)
                    finally
                        GlobalUnlock(hData) |> ignore
                else
                    ""
            else
                ""
        finally
            CloseClipboard() |> ignore

    let toClipboard (s:string) =
        try
            if (OpenClipboard(0n)) then
                EmptyClipboard() |> ignore  // this line was critical if you actually want to put data on the clipboard!!!!
                let hData = System.Runtime.InteropServices.Marshal.StringToHGlobalUni(s)
                SetClipboardData(13u(*CF_UNICODETEXT*),hData) |> ignore
        finally
            CloseClipboard() |> ignore

    let toClipboardAppendNL s = toClipboard (s+"\r\n")
    let splitToLines (s:string) = s.Split('\r','\n') |> Seq.filter (System.String.IsNullOrWhiteSpace >> not)
