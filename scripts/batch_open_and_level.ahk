; Path to Gwyddion executable
gwyddionPath := "C:\Program Files\Gwyddion\bin\gwyddion.exe"

; Folder containing the TIFF files
tiffFolder := "D:\Career\physics\M2\INSP Internship\data\ans\22R033-20250509\temp"

Loop, Files, %tiffFolder%\*.tiff
{
    fullPath := A_LoopFileFullPath
    fileName := A_LoopFileName
    fileNameNoExt := RegExReplace(fileName, "\..*$") ; Strip extension

    ; Run Gwyddion with current file
    Run, "%gwyddionPath%" "%fullPath%"
    WinWait, Gwyddion
    IfWinNotActive, Gwyddion, , WinActivate
    WinWaitActive, Gwyddion
    Sleep, 2000

    ; Activate Gwyddion main window
    WinActivate, ahk_class Gwyddion

    ; Activate Gwyddion main window
    WinActivate, ahk_class Gwyddion

    ; Apply Plane Level
    Send, !d
    Sleep, 300
    Send, l
    Sleep, 300
    Send, l
    Sleep, 800

    ; Apply Align Rows
    Send, !d
    Sleep, 300
    Send, c
    Sleep, 500
    Send, a
    Sleep, 500
    Send, {Enter}
    Sleep, 500
    Send, {Enter}
    Sleep, 1000

    ; Close the unchanged copy
    Send, !{F4}
    Sleep, 800

    ; Focus the TIFF display window using dynamic match
    WinGet, id, List,,, Program Manager
    Loop, %id%
    {
        this_id := id%A_Index%
        WinGetTitle, this_title, ahk_id %this_id%
        IfInString, this_title, %fileNameNoExt%
        {
            WinActivate, ahk_id %this_id%
            break
        }
    }
    Sleep, 500

    ; Save the processed file
    Send, ^+s
    Sleep, 800
    Send, {Right}
    Sleep, 300
    Send, ^+{Right}
    Sleep, 300
    Send, {Backspace}
    Sleep, 300
    Send, .npy
    Sleep, 500
    Send, !s
    Sleep, 1000

    ; Close Gwyddion (double Alt+F4)
    WinActivate, ahk_class Gwyddion
    Sleep, 300
    Send, !{F4}
    Sleep, 300
    Send, !{F4}
    Sleep, 1000

    ; === Reset Desktop Focus ===
    ; Refocus desktop to ensure no weird leftovers
    DllCall("SwitchToThisWindow", "UInt", WinExist("ahk_class Progman"), "Int", 1)
    Sleep, 300

    ; Optional: Click empty desktop space to clear any selection
    CoordMode, Mouse, Screen
    MouseMove, 20, 20
    Click
    Sleep, 300
}

MsgBox, Done processing all TIFF files!
return
