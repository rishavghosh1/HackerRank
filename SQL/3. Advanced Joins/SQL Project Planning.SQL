https://www.hackerrank.com/challenges/sql-projects/problem
/*
OUTPUTS: start_date and end_dates for the project
ORDER By- (end_date-start_date),start_date
LOGIC- Start dates of the project will not coincide with the end dates and vice versa. 
*/

SELECT start_date, MIN(end_date)
FROM
    (SELECT start_date
     FROM projects
     WHERE start_date NOT IN (SELECT end_date FROM projects)
    )a,
    (SELECT end_date
     FROM projects
     WHERE end_date NOT IN (SELECT start_date FROM projects)
    )b
WHERE start_date < end_date
GROUP BY start_date
ORDER BY MIN(end_date)-start_date,start_date
