/*
OUTPUT- hacker_id, name, total score
ORDER BY- score DESC, hacker_id ASC
CONDITION- total score has to be >0
LOGIC- Find max scores of submissions grouped by challenge_id and hacker_id. Then we add in a contraint which looks at the sum total of scores and filters out those with a total of 0
LEARNINGS- include all selected columns in group by statement
*/


SELECT s.hacker_id, h.name, SUM(s.score1)
FROM(
SELECT s1. hacker_id, s1.challenge_id, max(s1.score) as score1
FROM submissions s1
GROUP BY s1.hacker_id, s1.challenge_id) s
JOIN hackers h ON s.hacker_id= h.hacker_id
GROUP BY s.hacker_id, h.name
HAVING SUM(s.score1) > 0
ORDER BY SUM(s.score1) DESC, s.hacker_id
