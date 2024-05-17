using Microsoft.ProgramSynthesis.Matching.Text;
using System.Collections.Generic;
using System;

Session session = new Session();

IEnumerable<String> inputs1 = new[] {
 "21-Feb-73",
 "2 January 1920a ",
 "4 July 1767 ",
 "1892",
 "11 August 1897 ",
 "11 November 1889 ",
 "9-Jul-01",
 "17-Sep-08",
 "10-May-35",
 "7-Jun-52",
 "24 July 1802 ",
 "25 April 1873 ",
 "24 August 1850 ",
 "Unknown ",
 "1058",
 "8 August 1876 ",
 "26 July 1165 ",
 "28 December 1843 ",
 "22-Jul-46",
 "17 January 1871 ",
 "17-Apr-38",
 "28 February 1812 ",
 "1903",
 "1915",
 "1854",
 "9 May 1828 ",
 "28-Jul-32",
 "25-Feb-16",
 "19-Feb-40",
 "10-Oct-50",
 "5 November 1880 ",
 "1928",
 "13-Feb-03",
 "8-Oct-43",
 "1445",
 "8 July 1859 ",
 "25-Apr-27",
 "25 November 1562 ",
 "2-Apr-10", };

IEnumerable<String> inputs2 = new[] {
"Ind-674-PRO",
"US-823-JUN",
"US-238-JUN",
"QUAL-47",
"QUAL-21",
"Zim-843-PRO",
"Eng-781-JUN",
"Aus-664-PRO",
"QUAL-88",
"Ind-473-JUN",
"usa_837",
"Eng-573-JUN",
"Zim-392-PRO",
"QUAL-10", };

session.Inputs.Add(inputs2);
IReadOnlyList<PatternInfo> patterns = session.LearnPatterns(); // Five patterns are returned corresponding to the formats "dd-MMM-yy", "dd MMMM yyyy ", "yyyy", "Unknown", and "2 January 1920a ".

foreach (var pattern in patterns)
{
    Console.WriteLine(pattern.Regex);
}

