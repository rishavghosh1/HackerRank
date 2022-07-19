/*
OUTPUT: hacker_id, name, total no. of challenges
SORT: no. of challenges desc, hacker_id
LOGIC: Find out max no. of challenges for each hacker, find the no. of distinct no. of challenges and filter the count to include only these nos.
*/

SELECT c.hacker_id, h.name, count(c.challenge_id) as count1
FROM challenges c
JOIN hackers h on c.hacker_id=h.hacker_id
GROUP BY c.hacker_id, h.name
HAVING count1 = 
    (SELECT count(c2.challenge_id) as maxcount
     FROM challenges c2
     GROUP BY c2.hacker_id
     ORDER BY maxcount DESC
     LIMIT 1
    )
    OR count1 IN
    ( SELECT c3.count2 FROM
    (SELECT count(*) as count2
               from challenges
               group by hacker_id) c3
         group by c3.count2
         having count(c3.count2) = 1)
ORDER BY count1 DESC, hacker_id
