--https://www.hackerrank.com/challenges/the-pads/

--There are 2 queries which need to be executed here. For the first one, we need to use concat to display result together. We need to order by names alphabetically.
--For the next query, we need to aggregate the occupations to find the total no. of individuals for each occupation group. We also require concat function to display the result in the desired output

SELECT CONCAT(NAME,'(',SUBSTR(OCCUPATION,1,1),')')
FROM OCCUPATIONS
ORDER BY NAME;

SELECT CONCAT('There are a total of ',COUNT(OCCUPATION),' ',LOWER(OCCUPATION),'s.')
FROM OCCUPATIONS
GROUP BY OCCUPATION
ORDER BY COUNT(OCCUPATION),OCCUPATION;
