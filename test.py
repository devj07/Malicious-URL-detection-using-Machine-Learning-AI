product_list=[]
product_price=[]
sum=0
def EnterProduct():
    p_name=input("Enter Product Name:")
    p_price=float(input("Enter Product Price:"))
    product_list.append(p_name)
    if p_price>0:
        product_price.append(p_price)
    


while(True):
    ch=int(input("Press 1-Enter Product details,0-exit:"))
    if(ch==1):
        EnterProduct()
    if ch==0:
        break



for i in range (len(product_price)):
    sum+=product_price[i]

print(f"You have to pay a total amount of ${sum}")
print("Product Name " + " Product Price")
print("---------------------------------")
for j in range (len(product_list)):
    print(product_list[j] , "%.2f" %product_price[j])


# ch=input("Enetr 1-Enter product , 2-exit ")
# switch(ch):
#     if ch==1:
#         EnterProduct()
#     if ch==2:
#         break;



