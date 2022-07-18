/*
OUTPUT: 4 different columns- based on Occupation. 
ORDER BY: name
https://www.hackerrank.com/challenges/occupations/

LOGIC: We use rank function to sort and rank data based on occupation partition. Then we group using rank so that we get a table. Every derived table should have an alias. Can use multiple case when in order to pivot to different tables
*/

select min(Doctor), min(Professor), min(Singer), min(Actor)
from
(Select  RANK() OVER(PARTITION BY occupation ORDER BY name) rnk,
    case OCCUPATION when 'Doctor' then NAME end AS Doctor,
    case OCCUPATION when 'Professor' then NAME end AS Professor,
    case OCCUPATION when 'Singer' then NAME end AS Singer,
    case OCCUPATION when 'Actor' then NAME end AS Actor
from occupations) temp
group by rnk
order by rnk;
