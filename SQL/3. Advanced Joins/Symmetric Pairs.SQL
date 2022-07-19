/*
https://www.hackerrank.com/challenges/symmetric-pairs/problem
LOGIC: A.x = B.y and vice versa. Add constraint to make sure you include same points which are repeated as they are symmetric pairs. Also add a constraint X < Y
*/

SELECT A.X, A.Y 
FROM functions A 
JOIN functions B ON A.X=B.Y AND A.Y=B.X
GROUP BY A.X, A.Y
HAVING COUNT(A.X)>1 or A.X<A.Y
ORDER BY A.X 
