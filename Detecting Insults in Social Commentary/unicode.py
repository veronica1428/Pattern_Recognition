import unicodedata
import string
import re

s="http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python veronica sharma"
#s= "Gabriel.\\xc2 \\xc2\\xa0Getting time to blow that horn?"
#s="Oops! \xa0How embarrassing (for me)... you are right People1st... I bow to the superior movie intellect of your son :)"
#s="@Rourke needs to give me a fucking like\xa0\xa0@jwoude23\xa0hows everyone doin?"
#s="I'm amused by the fact that WeBlog cons drop any issue once it turns into a loser.\n\nThis is an expression of that amusement."
#s="C\xe1c b\u1ea1n xu\u1ed1ng \u0111\u01b0\u1eddng bi\u1ec3u t\xecnh 2011 c\xf3 \xf4n ho\xe0 kh\xf4ng ? \nC\xe1c ng\u01b0 d\xe2n ng\u1ed3i cu\xed \u0111\u1ea7u chi\u1ee5 nh\u1ee5c c\xf3 \xf4n ho\xe0 kh\xf4ng ?\nC\xe1c n\xf4ng d\xe2n gi\u1eef \u0111\u1ea5t \u1edf V\u0103n Giang, C\u1ea7n Th\u01a1 c\xf3 \xf4n ho\xe0 kh\xf4ng ?\n.................\nR\u1ed1t cu\u1ed9c \u0111\u01b0\u1ee3c g\xec\xa0 th\xec ch\xfang ta \u0111\xe3 bi\u1ebft !\nAi c\u0169ng y\xeau chu\u1ed9ng ho\xe0 b\xecnh, nh\u01b0ng \u0111\xf4i khi ho\xe0 b\xecnh ch\u1ec9 th\u1eadt s\u1ef1 \u0111\u1ebfn sau chi\u1ebfn tranh m\xe0 th\xf4i.\nKh\xf4ng c\xf2n con \u0111\u01b0\u1eddng n\xe0o ch\u1ecdn kh\xe1c \u0111\xe2u, \u0111\u1eebng m\u01a1 th\xeam n\u01b0\xe3."
#s="@xxcorey92\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 im fucking with you that makes my dick hard\xa0 and you know what i do...i let them catch me looking at them then quickly pretend to look away and not to see them\xa0 to see there reaction ololol"
#s= "\xc2the majority"
#s=s.strip()
#s=s.replace("\n", " ")
#s= s.encode("ascii","replace")
#s= "C\xe1c b\u1ea1n xu\u1ed1ng \u0111\u01b0\u1eddng bi\u1ec3u t\xecnh 2011 c\xf3 \xf4n ho\xe0 kh\xf4ng ? \nC\xe1c ng\u01b0 d\xe2n ng\u1ed3i cu\xed \u0111\u1ea7u chi\u1ee5 nh\u1ee5c c\xf3 \xf4n ho\xe0 kh\xf4ng ?\nC\xe1c n\xf4ng d\xe2n gi\u1eef \u0111\u1ea5t \u1edf V\u0103n Giang, C\u1ea7n Th\u01a1 c\xf3 \xf4n ho\xe0 kh\xf4ng ?\n.................\nR\u1ed1t cu\u1ed9c \u0111\u01b0\u1ee3c g\xec\xa0 th\xec ch\xfang ta \u0111\xe3 bi\u1ebft !\nAi c\u0169ng y\xeau chu\u1ed9ng ho\xe0 b\xecnh, nh\u01b0ng \u0111\xf4i khi ho\xe0 b\xecnh ch\u1ec9 th\u1eadt s\u1ef1 \u0111\u1ebfn sau chi\u1ebfn tranh m\xe0 th\xf4i.\nKh\xf4ng c\xf2n con \u0111\u01b0\u1eddng n\xe0o ch\u1ecdn kh\xe1c \u0111\xe2u, \u0111\u1eebng m\u01a1 th\xeam n\u01b0\xe3."

#s= s.replace('\\\\', '\\')
#s= s.encode().decode('unicode_escape')
#s= "m//y \\xcaelmn"
#s=s.replace('//','/')
#s=s.decode('unicode_escape').encode('utf8')

#s=s.decode('utf-8', 'ignore').encode('ascii').decode('unicode_escape')
#s=s.encode().decode('unicode_escape').encode('latin1',errors='ignore').decode('utf-8')
#s=unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')

#s=filter(lambda x:x in string.printable, s)
#s=s.decode('unicode_escape').encode('utf8')

#s= s.replace(s.encode().decode('unicode_escape'))
#s= s.encode("ascii","ignore")

URLless_string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', s)
#print URLless_string

print(s)
