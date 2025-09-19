-- select s.store_name, count(*)
-- from sales.stores s join sales.orders o
-- on s.store_id = o.store_id
-- group by store_name
-- having count(*) > 300
-- order by count(*)

-- select product_name, list_price 
-- from production.products
-- order by list_price desc
-- limit 5 -- gives me just 5 of this query(same like top in mysql)

-- select product_name, list_price 
-- from production.products
-- order by list_price desc
-- -- this gives me first 5 rows and if list_price(which is in ordered by) is present in the 5th row and there's more of it, it gets it also
-- fetch first 5 rows with ties

-- select product_name, list_price 
-- from production.products
-- where list_price = (select max(list_price) from production.products);


