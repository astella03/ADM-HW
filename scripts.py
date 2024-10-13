# Say "Hello, World!" With Python
print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if n % 2 != 0 :
    print("Weird")
else:
    if n in range(2, 6):
        print("Not Weird")
    if n in range(6, 21):
        print("Weird")
    if n > 20:
        print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())
for i in range(0, n):
    print(i**2)

# Print Function
if __name__ == '__main__':
    n = int(input())
stringa=""
for i in range(1,n+1):
  stringa+=f"{i}"
print(stringa)

# Write a function
def is_leap(year):
    leap = False
    
    if year % 4 == 0:
        if year % 100 != 0:
            leap=True
        if year % 100 == 0 and year % 400 == 0:
            leap=True
        if year % 100 == 0 and year % 400 != 0:
            leap=False
    
    return leap

# sWAP cASE
def swap_case(s):
    new_s = ""
    
    for i in s:
        if i.islower()==True:
            new_s += i.upper()
        else:
            new_s += i.lower()
    return new_s

# String Split and Join
def split_and_join(line):
    # write your code here
    line = line.split(" ")
    line = "-".join(line)
    return line

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    string = string[:position] + character + string[position+1:]
    return string

# Find a string
def count_substring(string, sub_string):
    count=0
    for i in range(0,len(string)+1):
        sub = string[i:len(sub_string)+i]
        if sub in string and sub==sub_string:
            count+=1
    return count

# String Validators
if __name__ == '__main__':
    s = input()
    
print(any(i.isalnum() for i in s))
print(any(i.isalpha() for i in s))
print(any(i.isdigit() for i in s))
print(any(i.islower() for i in s))
print(any(i.isupper() for i in s))

# Text Alignment
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

def wrap(string, max_width):
    l = list(textwrap.wrap(string, max_width))
    s = ""
    for i in l:
        s+=i+"\n"
    return s

# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, M = map(int, input().split())
N_half=int((int(N)/2)+0.5)
if N>5 and N<101 and N%2!=0 and M>15 and M<303 and M==N*3:
  count=1
  for i in range(1,N_half):
    print((".|."*count).center(M,"-"))
    count+=2
  print(("WELCOME").center(M,"-"))
  count=count-2
  for i in range(N_half,N):
    print((".|."*count).center(M,"-"))
    count=count-2

# String Formatting
def print_formatted(number):
    # your code goes here
    width = len(bin(number)) - 2  
    for i in range(1, number + 1):
        print(f"{i:{width}d} {i:{width}o} {i:{width}X} {i:{width}b}")

# Alphabet Rangoli
alphabet = {}
for i in range(ord("a"),ord("z")+1):
  alphabet[i-ord("a") +1]=chr(i)
def print_rangoli(size):
    # your code goes here
    lista = []
    first = alphabet[size]
    second = alphabet[size]
    lista.append(first.center(size*4 -3 ,"-"))
    for i in range(1,size):
      second += "-"+ alphabet[size-i]
      lista.append((second+"-"+first[::-1]).center(size*4 - 3,"-"))
      first = second
    
    for i in lista[:size]:
      print(i)
    
    for i in lista[-2::-1]:
      print(i)

# Capitalize!

# Complete the solve function below.
def solve(s):
    return " ".join([elem.capitalize() for elem in s.split(" ")])

# The Minion Game
def minion_game(string):
    # your code goes here
    v = 'AEIOU'
    s_l = len(string)
    kevin_score, stuart_score = 0, 0
    for i in range(s_l):
        if s[i] in v:
            kevin_score += (s_l - i)
        else:
            stuart_score += (s_l - i)
    if kevin_score > stuart_score:
        print("Kevin", kevin_score)
    elif kevin_score < stuart_score:
        print("Stuart", stuart_score)
    else:
        print("Draw")


# Introduction to Sets
def average(array):
    # your code goes here
    avg = sum(set(array)) / len(set(array))
    return avg

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
l = []
for a in range(x+1):
  for b in range(y+1):
    for c in range(z+1):
      if a+b+c!=n:
        l.append([a,b,c])
print(l)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
arr = list(arr)
arr = list(dict.fromkeys(arr))
arr.sort(reverse=False)
print(arr[-2])

# Nested Lists
scores = {}
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        scores[name]=score
runn = sorted(set(scores.values()))[1]
students = sorted([student for student,score in scores.items() if scores[student]==runn])
for i in students:
  print(i)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
mean = round(sum(student_marks[query_name])/len(student_marks[query_name]),2)
print("{:.2f}".format(mean))

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    
    t = tuple(integer_list)
    print(hash(t))

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    N = int(input())
    l=[]
    for _ in range(N):
        country = input()
        l.append(country)
        
print(len(set(l)))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    command = list(map(str, input().split()))
    if command[0] == "pop":
      s.pop()
    elif command[0] == "remove":
      s.remove(int(command[1]))
    elif command[0] == "discard":
      s.discard(int(command[1]))
print(sum(s))

# Set .union() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
roll = set(map(int, input().split()))
b = int(input())
num = set(map(int, input().split()))
print(len(roll | num))
    

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
roll = set(map(int, input().split()))
b = int(input())
num = set(map(int, input().split()))
print(len(roll & num))
    

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
roll = set(map(int, input().split()))
b = int(input())
num = set(map(int, input().split()))
union = roll|num
print(
    len(union-num)
)
    

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
roll = set(map(int, input().split()))
b = int(input())
num = set(map(int, input().split()))
print(len(roll ^ num))
    

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_A = int(input())
A = set(map(int,input().split()))
N = int(input())
for _ in range(N):
    operation = list(map(str, input().split()))
    other = set(map(int, input().split()))
    if operation[0]=="update":
        A.update(other)
    elif operation[0]=="intersection_update":
        A.intersection_update(other)
    elif operation[0]=="difference_update":
        A.difference_update(other)
    elif operation[0]=="symmetric_difference_update":
        A.symmetric_difference_update(other)
print(sum(A))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
K = int(input())
room = list(map(int, input().split()))
room_num = sum(room)
room_num_sum = sum(set(room)) * K
captain = (room_num_sum - room_num) // (K - 1)
print(captain)

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for _ in range(T):
    elem_A=int(input())
    A=set(map(int, input().split()))
    elem_B = int(input())
    B=set(map(int, input().split()))
    
    if A.issubset(B):
        print(True)
    else:
        print(False)

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
other_tot = set()
for _ in range(n):
    other = set(map(int, input().split()))
    other_tot = other_tot | other
if A.issuperset(other_tot):
    print(True)
else:
    print(False)
    

# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
elem_M = set(map(int, input().split()))
N = int(input())
elem_N = set(map(int, input().split()))
diff = elem_M.difference(elem_N) | elem_N.difference(elem_M)
for i in sorted(diff):
    print(i)

# Lists
n = int(input())
l=[]
for _ in range(n):
    cmd = list(map(str, input().split()))
    
    if cmd[0]=="insert":
        l.insert(int(cmd[1]),int(cmd[2]))
    if cmd[0]=="print":
        print(l)  
    if cmd[0]=="remove":
        l.remove(int(cmd[1]))
    if cmd[0]=="append":
        l.append(int(cmd[1]))
    if cmd[0]=="sort":
        l.sort()
    if cmd[0]=="pop":
        l.pop()
    if cmd[0]=="reverse":
        l.reverse()  

# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
date = list(map(int, input().split()))
days = dict((enumerate(calendar.day_name)))
index = calendar.weekday(date[2], date[0], date[1])
print(days[index].upper())

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime
# Complete the time_delta function below.
def time_delta(t1, t2):
    format_ = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, format_)
    t2 = datetime.strptime(t2, format_)
    return str(int(abs((t1-t2).total_seconds()))) 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
x = int(input())
sizes = list(map(int, input().split()))
n = int(input())
disp = Counter(sizes)
earn = 0
for _ in range(n):
    customer = list(map(int, input().split()))
    if disp[customer[0]]>0:
        earn+=customer[1]
        disp[customer[0]]-=1
print(earn)

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
d = deque()
n = int(input())
for _ in range(n):
    cmd = list(map(str, input().split()))
    if cmd[0]=="append":
        d.append(int(cmd[1]))
    if cmd[0]=="appendleft":
        d.appendleft(int(cmd[1]))
    if cmd[0]=="pop":
        d.pop()
    if cmd[0]=="popleft":
        d.popleft()
print(" ".join(map(str, d)))

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
n = int(input())
cols = list(map(str, input().split()))
cols = ",".join(map(str, cols))
l = list(cols.split())
Student = namedtuple('Student',cols)
#stu = Student()
average = 0 
for _ in range(n):
    inp = list(map(str, input().split()))
    stud = Student(inp[0],inp[1],inp[2],inp[3])
    average += int(stud.MARKS)
print(average/n)

# DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
d = defaultdict(list)
n,m = map(int, input().split())
for i in range(n):
    a = input()
    d[a].append(i+1)
for i in range(m):
    b = input()
    if b in d:
        print(*d[b])
    else:
        print(-1)

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
n = int(input())
d = OrderedDict()
for _ in range(n):
    item_name, price = map(str, input().rsplit(" ",1))
    if item_name in d.keys():
        d[item_name]+=int(price)
    else:
        d[item_name]=int(price)
for k,v in d.items():
    print(k,v)

# Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
l = []
n = int(input())
for _ in range(n):
    word = input()
    l.append(word)
new = Counter(l)
print(len(new.keys()))
print(*new.values())

# Company Logo
#!/bin/python3
import math
import os
import random
import re
import sys
from collections import Counter

if __name__ == '__main__':
    s = input()
new = Counter(list(sorted(s)))
sorted_new = sorted(new.items(), key=lambda x:x[1], 
reverse=True)
for i in sorted_new[:3]:
    print(*i)

# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
res = []
t = int(input())
for _ in range(t):
    n = int(input())
    sl = list(map(int, input().split()))
    for _ in range(n-1):
        if sl[0] >= sl[len(sl)-1]:
            a = sl[0]
            sl.pop(0)
        elif sl[0] < sl[len(sl)-1]:
            a = sl[len(sl)-1]
            sl.pop(len(sl)-1)
        else:
            pass
        if len(sl) == 1:
            res.append("Yes")
        if((sl[0] > a) or (sl[len(sl)-1] > a)):
            res.append("No")
            break
print("\n".join(res))

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for _ in range(T):
    a,b = map(str,input().split())
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as v:
      print("Error Code:", v)

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n,x = map(int,input().split())
l = []
def convert(value):
    try:
        return int(value) 
    except ValueError:
        return float(value)
for _ in range(x):
    marks = list(map(convert, input().split()))
    l.append(marks)
z = list(zip(*l))
for i in range(len(z)):
    print("{:.1f}".format(sum(z[i])/len(z[i])))

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    
arr.sort(key= lambda x:x[k])
for i in range(len(arr)):
  print(*arr[i])
  
  
  

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
s = input()
s = list(sorted(s))
new = ""
upper = "".join([x for x in s if x.isupper()])
lower = "".join([x for x in s if x.islower()])
num = "".join(sorted([x for x in s if x.isnumeric()], key=lambda x: int(x)%2==0))
new += lower + upper + num
print(new)

# Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    fibo = []
    if n==1:
        fibo.append(n-1)
    elif n>1:
        fibo = [0,1]
        i=0
        while len(fibo)!=n:
            first, second = fibo[i], fibo[i+1]
            third = second+first
            fibo.append(third)
            i+=1
    return fibo

# Arrays

def arrays(arr):
    # complete this function
    # use numpy.array
    arr.reverse()
    num = numpy.array(arr,float)
    return num

# Zeros and Ones
import numpy
n = list(map(int, input().split()))
print(numpy.zeros(tuple(n), dtype = numpy.int64))        
print(numpy.ones(tuple(n), dtype = numpy.int64))

# Linear Algebra
import numpy as np
arr = np.array([input().split() for _ in range(int(input()))], float)
print(round(np.linalg.det(arr),2))

# Shape and Reshape
import numpy as np
arr = np.array([input().split()], int)
print(np.reshape(arr,(3,3)))

# Concatenate
import numpy as np
n, m, p = map(int, input().split())
arr = np.array([ input().split()  for i in range(n+m)], int)
print(arr)

# Transpose and Flatten
import numpy as np
n, m = map(int, input().split())
A = np.array([ input().split() for i in range(n) ], int)
print(np.transpose(A))
print(A.flatten())


# Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
print(np.eye(n,m))


# Dot and Cross
import numpy as np
n = int(input())
A = np.array([ input().split() for i in range(n) ], int)
B = np.array([ input().split() for i in range(n) ], int)
print(np.dot(A,B))


# Min and Max
import numpy as np
n, m = map(int, input().split())
arr = np.array([ input().split() for i in range(n) ], int)
print(
    max(
        np.min(arr, axis=1)
    )
)

# Sum and Prod
import numpy as np
n, m = map(int, input().split())
arr = np.array([ input().split() for i in range(n) ], int)
print(np.prod(np.sum(arr, axis=0)))

# Inner and Outer
import numpy as np
A = np.array( input().split() , int)
B = np.array( input().split() , int)
print(np.inner(A,B))
print(np.outer(A,B))


# Array Mathematics
import numpy as np
n, m = map(int, input().split())
A = np.array([input().split() for i in range(n)], int)
B = np.array([input().split() for i in range(n)], int)
print(np.add(A,B))
print(np.subtract(A,B))
print(np.multiply(A,B))
print(np.floor_divide(A,B))
print(np.mod(A,B))
print(np.power(A,B))


# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(sign=' ')
A = np.array(input().split(), float)
print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))


# Polynomials
import numpy as np
p = np.array(input().split(), float)
x = float(input())
print(np.polyval(p, x))


# Mean, Var, and Std
import numpy as np
n, m = map(int, input().split())
arr = np.array([ input().split() for i in range(n) ], int)
print(np.mean(arr, axis=1))
print(np.var(arr, axis=0))
print(round(np.std(arr, axis=None),11))

# XML 1 - Find the Score

def get_attr_number(node):
    # your code goes here
    return len(node.attrib) + sum(get_attr_number(child) for child in node);

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level + 1)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # complete the function
        f(['+91 ' + i[-10:-5] + ' ' + i[-5:] for i in l])
    return fun

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        # complete the function
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
from collections import Counter
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#
def birthdayCakeCandles(candles):
    # Write your code here
    c = Counter(candles)
    return c[max(c.keys())]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    # Write your code here
    if x2>x1 and v2>=v1:
        return "NO"
    if x1>x2 and v1>=v2:
        return "NO"
    if abs(x1-x2)%abs(v1-v2)==0:
        return "YES"
    else:
        return "NO"
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    # Write your code here
    liked = 2
    total = liked
    for _ in range(n-1):
        shared = liked*3
        liked = shared//2
        total += liked
    return total
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
  # Write your code here
  n = sum([int(i) for i in n])
  p = n*k
  while int(p)>10:
      l = [int(i) for i in str(p)]
      p = sum(l)
  return p
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    # Write your code here
    store = arr[n-1]
    arr.pop()
    for i in range((len(arr)-1),-1,-1):
        if arr[i]>store:
            arr.insert(i,arr[i])
            print(*arr)
            arr.remove(arr[i])
        else:
            arr.insert(i+1,store)
            print(*arr)
            break
            
    if arr[0]>store:
        arr.insert(0,store)
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    # Write your code here
    for i in range(1, n):
        x = arr[i]
        y = i-1
        while y >= 0 and arr[y] > x:
            arr[y+1] = arr[y]
            y -= 1
        arr[y+1] = x
        print(' '.join(map(str, arr)))
        
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

# Merge the Tools!
from collections import OrderedDict
def merge_the_tools(string, k):
    # your code goes here
    for i in range(0, len(string), k):
        print(''.join(OrderedDict.fromkeys(string[i:i + k])))


# No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = input().split()
array = input().split()
A = set(input().split())
B = set(input().split())
print(sum([(i in A) - (i in B) for i in array]))

# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
from re import match, compile
pattern = compile('^[-+]?[0-9]*\.[0-9]+$')
for _ in range(int(input())):
    print(bool(pattern.match(input())))

# Re.split()
regex_pattern = r'[.,]+'

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
m = re.search(r'([a-zA-Z0-9])\1', input().strip())
print(m.group(1) if m else -1)

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
v = 'aeiou'
c = 'qwrtypsdfghjklzxcvbnm'
match = re.findall(r'(?<=[' + c + '])([' + v + ']{2,})(?=[' + c + '])', input(), flags=re.I)
print('\n'.join(match or ['-1']))

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
subs = input()
ptrn = re.compile(subs)
m = ptrn.search(s)
if not m: print('(-1, -1)')
while m:
    print('({0}, {1})'.format(m.start(), m.end() - 1))
    m = ptrn.search(s, m.start() + 1)

# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for _ in range(int(input())):
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', lambda x: 'and' if x.group() == '&&' else 'or', input()))

# Validating Roman Numerals
import re
regex_pattern = r'M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$'

# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
[print('YES' if re.match(r'[789]\d{9}$', input()) else 'NO') for _ in range(int(input()))]

# Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
ptrn = r'^<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>$'
for _ in range(int(input())):
    n, e = input().split(' ')
    if re.match(ptrn, e):
        print(n, e)

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for _ in range(int(input())):
    m = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if m:
        print(*m, sep='\n')

# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])
    def handle_endtag(self, tag):
        print ('End   :', tag)
    def handle_startendtag(self, tag, attrs):
        print ('Empty :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])

parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, comment):
        if '\n' in comment:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print(comment)
    def handle_data(self, data):
        if data == '\n': return
        print('>>> Data')
        print(data)
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]

html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')

# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
pattern = re.compile(
    r'^'
    r'(?!.*(\d)(-?\1){3})'
    r'[456]\d{3}'
    r'(?:-?\d{4}){3}'
    r'$')
for _ in range(int(input().strip())):
    print('Valid' if pattern.search(input().strip()) else 'Invalid')

# Validating Postal Codes
regex_integer_in_range = r'^[1-9][\d]{5}$'  # Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r'(\d)(?=\d\1)'  # Do not delete 'r'.

# Matrix Script
#!/bin/python3
import math
import os
import random
import re
import sys
n, m = map(int, input().split())
a, b = [], ''
for _ in range(n):
    a.append(input())
for z in zip(*a):
    b += ''.join(z)
print(re.sub(r'(?<=\w)([^\w]+)(?=\w)', ' ', b))

