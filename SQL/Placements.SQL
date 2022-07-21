/*
OUTPUT: s.name
ORDER BY: Friend's salary
LOGIC: First we have to compare student salary with their friend's salary. Join packages first as studen'ts salary and then as friend's salary
*/

SELECT name
FROM(
SELECT f1.id, p1.salary as salary, p2.salary as fsalary
FROM friends f1 JOIN packages p1 ON f1.id= p1.id
JOIN packages p2 ON f1.friend_id= p2.id
WHERE p1.salary < p2.salary) f
JOIN students s
ON f.id= s.id
ORDER BY f.fsalary

