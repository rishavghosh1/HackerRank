https://www.hackerrank.com/challenges/binary-search-tree-1/problem?isFullScreen=true

/*
PRINT   Root if P=0
        Leaf if it it not present in P
        Inner if common in N and P
ORDER BY NODE VALUE

*/

SELECT b.N,
       (CASE WHEN b.P IS NULL
             THEN 'Root' 
             WHEN EXISTS (SELECT 1 FROM BST b2 WHERE b2.P = b.N) 
             THEN 'Inner'
             ELSE 'Leaf'
        END)
FROM bst b 
ORDER BY N;
