/*
https://www.hackerrank.com/challenges/the-report/
OUTPUT: Name, grades, marks
SORT: Grade Descending, name ascending, marks ascending
LOGIC: join on marks being within the min and max range (on- between)
*/

SELECT CASE WHEN
    g.grade > 7 THEN s.name
    ELSE NULL
END as name, g.grade, s.marks
FROM students s
LEFT JOIN grades g
    ON s.marks BETWEEN g.min_mark and g.max_mark
ORDER BY g.grade DESC, s.name,s.marks
