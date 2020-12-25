Attribute VB_Name = "PNUpdate"
Sub upload_database()

Dim con As ADODB.Connection
Dim strCon As String
Dim sSQL As Variant
Dim rs As ADODB.Recordset

Dim i As Integer
Dim j As Integer
Dim k As Integer

Dim T_NAME As String
Dim T_AGE As Integer
Dim T_GENDER As String
Dim T_NOTES As String

Dim TRIAL_NAME As String

Dim H_AUTO_KEY As Variant
Dim T_AUTO_KEY As Variant

Dim wb As Workbook
Set wb = ActiveWorkbook

Set con = New Connection

T_NAME = wb.Sheets("Sheet1").Cells(4, 3).Value
T_AGE = wb.Sheets("Sheet1").Cells(5, 3).Value
T_GENDER = wb.Sheets("Sheet1").Cells(6, 3).Value
T_NOTES = wb.Sheets("Sheet1").Cells(7, 3).Value

If Header_Duplicate(T_NAME, T_AGE, T_GENDER, T_NOTES) = "T" Then
    GoTo duplicate_header
End If


strCon = "Provider=SQLOLEDB;" & _
         "Data Source=RAC-DESKTOP\SQLEXPRESS;" & _
         "Initial Catalog=PNLab;" & _
         "Integrated Security=SSPI;"
         
con.Open strCon

sSQL = "insert into data_headers(T_NAME,T_AGE,T_GENDER,T_NOTES) values('"
sSQL = sSQL & T_NAME & "',"
sSQL = sSQL & T_AGE & ",'"
sSQL = sSQL & T_GENDER & "','"
sSQL = sSQL & T_NOTES & "')"

con.Execute sSQL
Set rs = con.Execute("SELECT @@Identity")
H_AUTO_KEY = rs.Fields(0).Value

For i = 1 To 11

    TRIAL_NAME = wb.Sheets(i).Cells(9, 2).Value
    If wb.Sheets(i).Cells(14, 4).Value <> "" Then
        con.Execute ("insert into data_trials(TRIAL_NAME,T1_C,T2_C,T3_C,T4_C,T5_C,T6_C,T7_C,T8_C,T1_I,T2_I,T3_I,T4_I,T5_I,T6_I,T7_I,T8_I,T1_S,T2_S,T3_S,T4_S,T5_S,T6_S,T7_S,T8_S,H_AUTO_KEY) values('" & TRIAL_NAME & "','" & _
        wb.Sheets(i).Cells(13, 4).Value & "','" & _
        wb.Sheets(i).Cells(13, 5).Value & "','" & _
        wb.Sheets(i).Cells(13, 6).Value & "','" & _
        wb.Sheets(i).Cells(13, 7).Value & "','" & _
        wb.Sheets(i).Cells(13, 8).Value & "','" & _
        wb.Sheets(i).Cells(13, 9).Value & "','" & _
        wb.Sheets(i).Cells(13, 10).Value & "','" & _
        wb.Sheets(i).Cells(13, 11).Value & "','" & _
        wb.Sheets(i).Cells(14, 4).Value & "','" & _
        wb.Sheets(i).Cells(14, 5).Value & "','" & _
        wb.Sheets(i).Cells(14, 6).Value & "','" & _
        wb.Sheets(i).Cells(14, 7).Value & "','" & _
        wb.Sheets(i).Cells(14, 8).Value & "','" & _
        wb.Sheets(i).Cells(14, 9).Value & "','" & _
        wb.Sheets(i).Cells(14, 10).Value & "','" & _
        wb.Sheets(i).Cells(14, 11).Value & "'," & _
        wb.Sheets(i).Cells(15, 4).Value & "," & _
        wb.Sheets(i).Cells(15, 5).Value & "," & _
        wb.Sheets(i).Cells(15, 6).Value & "," & _
        wb.Sheets(i).Cells(15, 7).Value & "," & _
        wb.Sheets(i).Cells(15, 8).Value & "," & _
        wb.Sheets(i).Cells(15, 9).Value & "," & _
        wb.Sheets(i).Cells(15, 10).Value & "," & _
        wb.Sheets(i).Cells(15, 11).Value & "," & H_AUTO_KEY & ")")
        
        Set rs = con.Execute("SELECT @@Identity")
        T_AUTO_KEY = rs.Fields(0).Value
        
        For j = 1 To 8
            For k = 18 To 3017
                con.Execute ("insert into data_details(TRIAL_NUM,STAMP_ORDER,X_DATA,Y_DATA,TIME_DATA,H_AUTO_KEY,T_AUTO_KEY) values(" & _
                j & "," & _
                k - 17 & "," _
                & wb.Sheets(i).Cells(k, 13 + j * 3).Value & "," _
                & wb.Sheets(i).Cells(k, 14 + j * 3).Value & "," _
                & wb.Sheets(i).Cells(k, 15 + j * 3).Value & "," _
                & H_AUTO_KEY & "," _
                & T_AUTO_KEY & ")")
            Next k
        Next j
    End If
Next i

rs.Close
con.Close
Set rs = Nothing
Set con = Nothing

MsgBox ("Upload Complete")


Exit Sub

duplicate_header:
    MsgBox (T_NAME & " Already in Database")
End Sub

Function Header_Duplicate(NAME As String, AGE As Integer, GENDER As String, NOTES As String) As String

Dim con As ADODB.Connection
Dim rs As ADODB.Recordset
Dim strCon As String
Dim sSQL As String

Dim wb As Workbook

Set wb = ActiveWorkbook

Set con = New Connection

strCon = "Provider=SQLOLEDB;" & _
         "Data Source=RAC-DESKTOP\SQLEXPRESS;" & _
         "Initial Catalog=PNLab;" & _
         "Integrated Security=SSPI;"
         
con.Open strCon

sSQL = "select H_AUTO_KEY from DATA_HEADERS where T_NAME like '"
sSQL = sSQL & NAME & "' and T_AGE = "
sSQL = sSQL & AGE & " and T_GENDER = '"
sSQL = sSQL & GENDER & "' and T_NOTES like '"
sSQL = sSQL & NOTES & "'"
Set rs = con.Execute(sSQL)

If rs.EOF Then
    Header_Duplicate = "F"
Else
    Header_Duplicate = "T"
End If

rs.Close
con.Close
Set con = Nothing
Set rs = Nothing
End Function
