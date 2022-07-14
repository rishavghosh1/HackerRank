--Tables used:

--contests as c
--colleges as co
--challenges as ch
--view_stats
--submission_stats
--combined submission cs
--combined views cv

--TO PRINT:

--c.contest_id, c.hacker_id, name, and the sums of SUM(ss.total_submissions), SUM(ss.total_accepted_submissions), SUM(vs.total_views), and SUM(vs.total_unique_views)

--GROUP BY contest_id, hacker_id 
--ORDER by contest_id

WITH groupedviewstats as(
    SELECT challenge_id, SUM(total_views) as total_views, SUM(total_unique_views) as total_unique_views
    FROM views_stats GROUP BY challenge_id),
groupedsubmissionstats as(
    SELECT challenge_id, SUM(total_submissions) as total_submissions , SUM(total_accepted_submissions) as total_accepted_submissions
    FROM submission_stats GROUP BY challenge_id)

SELECT c.contest_id, c.hacker_id, ISNULL(SUM(total_submissions),0) total_submissions, 
ISNULL(SUM(total_accepted_submissions),0) total_accepted_submissions,
ISNULL(SUM(total_views),0) total_views, ISNULL(SUM(total_unique_views),0) total_unique_views
FROM contests c
INNER JOIN colleges co on c.contest_id=co.contest_id
INNER JOIN challenges ch on co.college_id=ch.college_id
LEFT JOIN groupedviewstats cv on ch.challenge_id=cv.challenge_id
LEFT JOIN groupedsubmissionstats cs on ch.challenge_id=cs.challenge_id
GROUP BY c.contest_id, c.hacker_id, c.name
HAVING SUM(total_views) >0 OR SUM(total_unique_views)>0  OR SUM(total_submissions)>0  OR SUM(total_accepted_submission) >0
ORDER BY c.contest_id;
