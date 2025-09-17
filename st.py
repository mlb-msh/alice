-- Q1: Hospital Patient Records Analysis

import System.IO

type Patient = (String, Int, Int)

-- Count patients by reason codes recursively
countReasons :: [Patient] -> (Int, Int, Int)
countReasons [] = (0,0,0)
countReasons ((_, _, reason):xs) =
    let (c1, c2, c3) = countReasons xs
    in case reason of
        1 -> (c1+1, c2, c3)
        2 -> (c1, c2+1, c3)
        3 -> (c1, c2, c3+1)
        _ -> (c1, c2, c3)

-- Count adults (18+)
countAdults :: [Patient] -> Int
countAdults [] = 0
countAdults ((_, age, _):xs)
    | age >= 18 = 1 + countAdults xs
    | otherwise = countAdults xs

-- Parse input line "Name:Age:Reason"
parsePatient :: String -> Patient
parsePatient str =
    let (n:a:r:_) = wordsWhen (==':') str
    in (n, read a, read r)

-- Split string utility
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s = case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
        where (w, s'') = break p s'

main :: IO ()
main = do
    putStrLn "Enter patient records (Name:Age:Reason). Blank line to stop:"
    input <- getContents
    let userData = filter (/= "") (lines input)
    let patients = if null userData
                      then [("Alice",25,1),("Bob",17,2),("Charlie",40,3),("Diana",30,1)]
                      else map parsePatient userData

    let (c1, c2, c3) = countReasons patients
    let adults = countAdults patients

    putStrLn $ "General Checkup: " ++ show c1
    putStrLn $ "Emergency: " ++ show c2
    putStrLn $ "Surgery: " ++ show c3
    putStrLn $ "Total Adults: " ++ show adults


{- 
Sample Input (if typing):
Alice:25:1
Bob:17:2
Charlie:40:3
Diana:30:1

(Or run without typing → uses same hardcoded data)
-}


-- Q2: Cinema Ticket Sales Report

import System.IO

type Sale = (String, Int)

-- Sum category tickets
sumCategory :: String -> [Sale] -> Int
sumCategory _ [] = 0
sumCategory cat ((c, q):xs)
    | cat == c  = q + sumCategory cat xs
    | otherwise = sumCategory cat xs

-- Calculate total revenue
revenue :: [Sale] -> Int
revenue [] = 0
revenue ((c, q):xs) =
    let price = case c of
                    "Adult"  -> 12
                    "Child"  -> 8
                    "Senior" -> 10
                    _        -> 0
    in q*price + revenue xs

-- Parse "Category:Quantity"
parseSale :: String -> Sale
parseSale str =
    let (c:q:_) = wordsWhen (==':') str
    in (c, read q)

-- String split utility
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s = case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
        where (w, s'') = break p s'

main :: IO ()
main = do
    putStrLn "Enter sales records (Category:Quantity). Blank line to stop:"
    input <- getContents
    let userData = filter (/= "") (lines input)
    let sales = if null userData
                   then [("Adult",3),("Child",5),("Senior",2),("Adult",2)]
                   else map parseSale userData

    let adultTotal  = sumCategory "Adult" sales
    let childTotal  = sumCategory "Child" sales
    let seniorTotal = sumCategory "Senior" sales
    let totalRev    = revenue sales

    putStrLn "---- Cinema Sales Report ----"
    putStrLn $ "Adults: " ++ show adultTotal
    putStrLn $ "Children: " ++ show childTotal
    putStrLn $ "Seniors: " ++ show seniorTotal
    putStrLn $ "Total Revenue: $" ++ show totalRev

{- 
Sample Input:
Adult:3
Child:5
Senior:2
Adult:2

(Or run without typing → uses same hardcoded data)
-}


-- Q3: Student Academic Performance Report

import System.IO

type Student = (String, Int)

-- Classify student with guards
classify :: Student -> (String, Int, String)
classify (name, mark)
    | mark < 40  = (name, mark, "Fail")
    | mark < 60  = (name, mark, "Pass")
    | mark < 80  = (name, mark, "Merit")
    | otherwise  = (name, mark, "Distinction")

-- Recursively classify all
classifyAll :: [Student] -> [(String, Int, String)]
classifyAll [] = []
classifyAll (s:xs) = classify s : classifyAll xs

-- Count passes (mark >= 40)
countPasses :: [Student] -> Int
countPasses [] = 0
countPasses ((_, mark):xs)
    | mark >= 40 = 1 + countPasses xs
    | otherwise  = countPasses xs

-- Parse "Name:Mark"
parseStudent :: String -> Student
parseStudent str =
    let (n:m:_) = wordsWhen (==':') str
    in (n, read m)

-- String split utility
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s = case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
        where (w, s'') = break p s'

main :: IO ()
main = do
    putStrLn "Enter student records (Name:Mark). Blank line to stop:"
    input <- getContents
    let userData = filter (/= "") (lines input)
    let students = if null userData
                      then [("Alice",85),("Bob",35),("Charlie",60),("Diana",75)]
                      else map parseStudent userData

    let results = classifyAll students
    let passCount = countPasses students

    putStrLn "---- Student Report ----"
    mapM_ print results
    putStrLn $ "Total Passed: " ++ show passCount

{- 
Sample Input:
Alice:85
Bob:35
Charlie:60
Diana:75

(Or run without typing → uses same hardcoded data)
-}
