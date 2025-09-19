-- select order_id, order_status, order_date, customer_id from sales.orders;

-- select customer_id, first_name, last_name from sales.customers
-- order by customer_id;

-- select first_name, last_name, email, order_id, order_date, store_id
-- from sales.customers c, sales.orders o
-- where c.customer_id = o.customer_id

-- select first_name, last_name, email, order_id, order_date, store_id
-- from sales.customers c join sales.orders o
-- on c.customer_id = o.customer_id

-- select first_name, last_name, email, phone, order_id, order_status, order_date
-- from sales.orders o, sales.staffs s
-- where o.staff_id = s.staff_id

-- select first_name, last_name, email, phone, order_id, order_status, order_date
-- from sales.orders o inner join sales.staffs s
-- on o.staff_id = s.staff_id

-- select first_name, last_name, email, phone, order_id, order_status, order_date
-- from sales.orders o right outer join sales.staffs s
-- on o.staff_id = s.staff_id

-- left outer join >> gives you all left even if it's null in the right table
-- same for right outer join
-- full outer join merges bet both of them

-- select c.customer_id, first_name, last_name, email, order_id, order_date
-- from sales.customers c left outer join sales.orders o
-- on c.customer_id = o.customer_id
-- here in select we must say c.customer_id or o.customer_id
-- because it's present in both tables

-- select first_name, last_name, o.order_id, order_date, store_name, s.store_id
-- from sales.customers c, sales.orders o, sales.stores s
-- where c.customer_id = o.customer_id and o.store_id = s.store_id

-- select first_name, last_name, o.order_id, order_date, store_name, s.store_id
-- from sales.customers c 
-- join sales.orders o 
-- on c.customer_id = o.customer_id
-- join sales.stores s
-- on o.store_id = s.store_id

-- select o.order_id, i.list_price, p.product_id, p.product_name
-- from sales.orders o, sales.order_items i, production.products p
-- where o.order_id = i.order_id 
--   and i.product_id = p.product_id

-- select first_name, last_name, brand_name 
-- from sales.customers c, sales.orders o, sales.order_items oi, production.products p, production.brands b
-- where c.customer_id = o.customer_id
-- and o.order_id = oi.order_id
-- and oi.product_id = p.product_id
-- and p.brand_id = b.brand_id

-- select c.first_name || ' ' ||  c.last_name as full_name, brand_name 
-- from sales.customers c, sales.orders o, sales.order_items oi, production.products p, production.brands b
-- where c.customer_id = o.customer_id
-- and o.order_id = oi.order_id
-- and oi.product_id = p.product_id
-- and p.brand_id = b.brand_id

-- select c.first_name || ' ' ||  c.last_name as full_name, brand_name 
-- from sales.customers c join sales.orders o on c.customer_id = o.customer_id
-- join sales.order_items oi on o.order_id = oi.order_id
-- join production.products p on oi.product_id = p.product_id
-- join production.brands b on p.brand_id = b.brand_id

--   aggregation functions

-- select max(list_price) from production.products

-- select avg(list_price) from production.products

-- select sum(list_price) as solyyyy from production.products

-- select sum(list_price) as "uhhhh" from production.products

-- select count(list_price) from production.products -- counts rows

-- select count(*) from production.products -- counts rows

-- select count(*), max(order_date), min(order_date) from sales.orders

-- select max(list_price) from production.products
-- where category_id = 3

select max(order_date) 
from sales.orders
where customer_id = 8711


