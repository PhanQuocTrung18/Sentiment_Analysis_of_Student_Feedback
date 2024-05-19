emotion2wordform_dict = { 
  	':)'  	:'colonsmile',
  	':('  	:'colonsad',
  	'@@'  	:'colonsurprise',
	'<3'  	:'colonlove',
	':d'  	:'colonsmilesmile',
	':3'  	:'coloncontemn',
	':v'  	:'colonbigsmile',
	':_'  	:'coloncc',
	':p'  	:'colonsmallsmile',
	'>>'  	:'coloncolon',
	':">' 	:'colonlovelove',
	'^^'  	:'colonhihi',
	':'   	:'doubledot',
	":'(" 	:'colonsadcolon',
	':’(' 	:'colonsadcolon',
	':@'  	:'colondoublesurprise',
	'v.v' 	:'vdotv',
	'...' 	:'dotdotdot',
	'/'   	:'fraction',
	'c#'  	:'cshrap',
  	'=)'  	:'coloneyesmile',
  	':3'  	: 'coloncontemn',
  	'^^'  	: 'colonhihi',
  	'c++' 	: 'cplusplus',
  	'&'   	: 'and',
  	'?'   	:  'dotthinking',
  	'3dot0'	: 'dotnumbers',
}

wordform2vnese_dict = {
        'colonsmile' 		: 'cười nhẹ',
        'colonsad'   		: 'buồn',
        'colonsurprise'		: 'ngạc nhiên',
	'colonlove'     	: 'trái tim',
	'colonsmilesmile'	: 'cười tươi',
	'coloncontemn'		: 'dễ thương',
	'colonbigsmile'		: 'cười nhạt',
	'coloncc'		: 'thích',
	'colonsmallsmile'	: 'vui vẻ',
	'coloncolon'		: 'tuyệt',
	'colonlovelove'		: 'thẹn thùng',
	'colonhihi'		: 'hạnh phúc',
	'doubledot'		: 'như là',
	'colonsadcolon'		: 'khóc',
	'colondoublesurprise': 'hơi ngạc nhiên',
	'vdotv'			: 'vân vân',
	'dotdotdot'		: 'vân vân',
	'fraction'		: 'chia',
	'cshrap'		: 'ngôn ngữ lập trình',
	'coloneyesmile'		: 'hạnh phúc',
	'cplusplus'		: 'ngôn ngữ lập trình',
	'and'			: 'và',
	'dotthinking'		: 'thắc mắc',
	'dotnumbers'		: 'chấm',
}


mispelling_dict = {

        'lặp đi lặp lại': ['lập đi lập lại','lập đi lặp lại','lặp đi lập lại','lap di lap lai'],
        'yêu cầu': ['yêu câu','yêu cau','yeu cầu','yeu câu','yeu cau'],        
        'kiến thức': ['kiến thực','kiến thưc','kiến thúc','kiến thuc','kiên thức','kiên thưc','kiên thúc','kiên thuc','kien thức','kien thưc','kien thúc','kien thuc'],              
        'bục giảng': ['bụt giảng', 'bục giãng','bụt giãng','buc giang','but giang','buc giảng','but giảng'],
        'tận tâm': ['tân tâm','tận tam','tân tam','tan tam'],
        'tăng cường': ['tăng cương','tăng cuong','tang cuong','tang cường','tang cương'],
        'trang thiết bị': ['trang thiet bi','trang thết bị','trang thêt bị','trang thiet bị','trang thiêt bị','trang thiêt bị','trang thiêt bi','trang thiêt bi','trang thiết bi','trang thết','trang thiết'],
        'thiết bị': ['thết bị','thêt bị','thêt bi','thet bị','thet bi','thiêt bị','thiêt bi','thiết bi','thiet bi','thiet bị'],
        'thống nhất cùng': ['thống nhất cũng'], 
        'khó khăn': ['khó khắn','khó khắn','khó khẵn','khó khặn','khó khan','kho khăn'],
        'phòng máy': ['phỏng máy','phong máy','phong may'],
        'khảo sát': ['khảo sat','khão sát','khão sat','khạo sát','khạo sat','khao sát','khảo sác','khảo sác','khảo sac','khao sat'],
        ' học sinh ': [' hs ',' học sin ',' hoc sin ','hoc sinh'],       
        'dễ hiểu': ['dễ hiêu','dê hieu','dê hiêu','de hieu'],
        'khó hiểu': ['khó hiêu','khó hieu','kho hiểu','kho hiêu','kho hieu'],     
        ' giảng dạy ': [' giảng dạ ', ' giản dạy ', ' giảng dạyy ',' giảng dậy ',' giang day '],       
        ' bình thường ': [' bình thưởng ', ' bình thườn ',' binh thuong '],
        'kéo dài' : ['kéo đài', 'keo dài', 'keo dai'],
        'bắt đầu' :['bắt đàu', 'bắt đau','bắt đâu','bắt dau','băt đầu','băt đâu','băt đau','băt dau','bat đầu','bat đâu','bat đau','bat dau'],
        'tìm hiểu' : ['tmf hiểu','tim hieu', 'timf hiểu','tìm hiều'],
        'áp đặt' : ['áp đặp', 'ap dat'],
        'lướt qua' : ['lước qua','lươt qua','luot qua'],
        'bổ sung' : ['bổ xung', 'bo sung', 'bô sung','bỏ sung'],
        ' giảng viên ' : [' giãng viên ',' giãng vien ',' giảng vien ',' giang viên ',' giản viên ',' giảng viến ',' giang vien ',' gv '],
        ' giáo viên ' : [' giá viên ',' giao viên ',' giao vien ',' giao viên ',' giáo vien '],
        'gần gũi': ['gần gủi','gần gui', 'gan gui'],
        'cộc lốc': ['cọc lốc','cọc lóc','coc loc'],      
        'vấn đề': ['vẫn đê','vân đê','vấn đê','vân đề','van de','van đề','vấn đe','vấn de'],
        ' mạnh dạn ': [' mạnh dạng '],     
        'nhiệt tình': ['nhiêt tình','nhiet tình','nhiệt tinh', 'nhiet tinh'],
        'bài mẫu': ['bày mẫu','bai mẫu','bai mau'],
        'truyền đạt': ['truyền đật','truyên dat','truyền đat','truyên đạt','truyên dạt','truyên đat','truyen dat'],      
        'trắc nghiệm': ['trăc nghiệm','trắc nghiem','trac nghiệm','trac nghiem'],
        'lập trình': ['lâp trình','lap trình','lap trinh'],
        'nhắc nhở': ['nhác nhở','nhac nho'],
        'lên lớp': ['lên lớn','len lop','lên lơp','lên lop'],
        'hoàn hảo': ['hoan hao'],
        'nhắn tin': ['nhăn tin','nhan tin'],
        'hứng thú': ['hướng thú','hung thu'],
        'cô dạy': ['cố dạy'],
        'hoàn chỉnh': ['hoàn chính','hoan chinh'],
        'tận tình': ['tân tình','tận hình','tận tinh','tan tinh'],
        'cuối cùng': ['cuối cung', 'cuoi cung'],
        'tiếp xúc': ['tiếp súc','tiếp xuc','tiep xúc','tiep xuc'],
        ' thực tế ': [' thức tế ',' thức tê ',' thự tế ',' thực tê ',' thực te ',' thưc tế ',' thưc tê ',' thưc te ',' thuc te '],
        'tâm huyết': ['tâm huyến','tâm huyêt','tam huyêt','tam huyết','tam huyet'],
        'công nghệ phần mềm': ['cnpm','cong nghe phan mem'],
        ' công nghệ ': [' côn nghệ ',' côn nghê ',' côn nghẹ ',' côn nghe ',' công nghê ',' công nghẹ ',' công nghe ',' cong nghệ ',' cong nghê ',' cong nghẹ ',' cong nghe '],
        'nâng cấp': ['năng cấp','nâng câp','nâng cap','nang câp','nang cấp','nang cap'],
        ' không thể ': [' không thẻ ',' ko thể ',' khum thể ',' k thể ',' không the ',' ko the ',' k the ',' khum the ',' khong the '],      
        'tham khảo': ['tam khảo','tham khao'],  
        'giám sát': ['dám sát','giám sat','giam sát','giam sat'],
        'một số' : ['mốt số','môt số','môt sô','môt so','mot số','mot sô','mot so'],       
        'hạn chế': ['hạn chê','hạn che','han che'],
        'cảm ơn' : ['cám ơn', 'cẻm ơn','kẽm ơn','cam on'],
        'ngắn gọn' : ['ngắn gọc','ngắn gọt','ngắn gõn','ngắn gôn','ngắn gon','ngan gon'],
        ' đảm bảo chất lượng ' : [' đbcl ',' dbcl '],
        'đảm bảo' : ['đảm bao','đảm bão','đãm bão','đãm bảo','đãm bao','đam bảo','đam bao','đam bão','dam bảo','dam bao','dam bão'],
        'hiện nay' : ['hien nay', 'hiên nay','hiện nây'],
        'quan tâm' : ['quan tấm', 'quan tam'],
        'nền tảng' : ['nền tẳng', 'nen tang'],
        'hàng ngày' : ['hằng ngày','hang ngay'],
        'nâng cao' : ['năng cao','nâng cáo', 'nâng cào', 'nâng cảo','nang cáo', 'nang cào', 'nang cảo','nang cao'],
        ' phòng đào tạo ' : [' pdt ',' pđt ','phòng đào tao','phòng dao tạo' ,'phong dao tao'],
        'đào tạo': ['đào tao','đào tảo','đào tão','đao tạo','đao tảo','đao tão','đao tao','dao tảo','dao dão','dao tạo','dao tao'],
        'khắt khe': ['khắc khe','khăc khe','khăt khe','khat khe'],
        'lưu loát': ['lưu lát','lưu loat','luu loát','luu loat','lưu lat','luu lat'],
        ' sinh viên ': [' sv ',' svien ',' sinhvien ',' sinh vien ',' sinh diên ',' sinh dien ',' sanh diên ',' sanh viên ',' sanh vien '],
        'dễ thương': ['dẽ thương','dẽ thuong','dễ thuong','dê thương','dê thuong','de thương','de thuong'],
        'vật lý': ['vậy lý','vậy lí','vật ly','vật li','vât ly','vât li','vat ly','vat li'],
        ' học kỳ ': [' hk ',' học ky ',' học ki ',' hoc ki ',' hoc ky ',' hoc kì ',' hoc kỳ '],
        ' học kỳ 1 ': [' hk1 ',' học ky 1 ',' học ki 1 ',' hoc ki 1 ',' hoc ky 1 ',' hoc kì 1 ',' hoc kỳ 1 '],
        ' học kỳ 2 ': [' hk2 ',' học ky 2 ',' học ki 2 ',' hoc ki 2 ',' hoc ky 2 ',' hoc kì 2 ',' hoc kỳ 2 '],
        'lười nhát': ['lười nhác','lười nhat','lười nhac','lươi nhát','lươi nhat','luoi nhat'],      
        ' bài tập ': [' bai tập ',' bài tâp ',' bài tap ',' bai tập ',' bai tâp ',' bai tap ',' btap ',' bt '],
        ' bài tập về nhà ': [' btvn '],
        ' ví dụ ': [' vd ',' vdu ',' vidu ',' ví du ',' vi dụ ',' vi du '],       
        'chất lượng': ['chất lương','chất luong','chât lượng','chât lương','chât luong','chat lượng','chat lương','chat luong'],     
        ' nội dung ': [' nối dung ',' nối dun ',' nội dun ',' nôi dung ',' nôi dun ',' noi dung '],
        'thực hành': ['thực hanh','thưc hành','thưc hanh','thuc hành','thuc hanh'],        
        'hướng dẫn': ['hướng dân','hướng dan','hương dẫn','hương dân','hương dan','huong dẫn','huong dân','huong dan'],
        'đầy đủ': ['đầy đu','đầy du','đây đủ','đây đu','đây du','đay đủ','đay đu','đay du','day đủ','day đu','day du'],
        'đúng giờ': ['đúng giơ','đúng gio','đung giờ','đung giơ ','đung gio','dung giờ','dung giơ','dung gio'],
        'thoải mái': ['thoải mai','thoai mái','thoai mai','thoãi mái','thoãi mai'],       
        'chi tiết': ['chi tiêt','chi tiet','chy tiết','chy tiêt','chy tiet','chỉ tiết'],
        'tóm tắt': ['tóm tăt','tóm tat','tom tắt','tom tăt','tom tat'],
        'hiền lành': ['hiền lanh','hiên lành','hiên lanh','hien lành','hien lanh'],
        'tận tụy': ['tận tuy','tân tụy','tân tuy','tan tụy','tan tuy'],
        ' vui vẻ ': [' vui ve ',' vui vee ',' vui vẻe ',' vui vẻee '],
        'hòa đồng': ['hòa đông','hòa đong','hòa dong','hoa đồng','hoa đông','hoa đong','hoa dong'],
        'thường xuyên': ['thường xuyen','thương xuyên','thương xuyen','thuong xuyên','thuong xuyen'],
        'khó chịu': ['khó chiu','kho chịu','khó chiệu','kho chiu'],
        'cơ sở': ['cơ sỏ','cơ sơ','co sở','co sỏ','co so','cơ sỡ','co sỡ'],
        'vật chất': ['vật chât','vật chát','vật chat','vât chất','vât chât','vât chát','vât chat','vat chất','vat chât','vat chát','vat chat'],
        'hấp dẫn': ['hấp dẩn','hấp dân','hấp dan','hâp dẫn','hâp dân','hâp dan','háp dẫn','háp dân','háp dan','hap dẫn','hap dân','hap dan'],
        ' trẻ trung ': [' tre trung ',' trẽ trung ',' trẻ trun ',' trẽ trun '],
        'dạy tệ': ['dạy tê','dạy te','day tệ','day tê','day te'],
        'quá tệ': ['quá tê','quá te','qua tệ','qua tê','qua te'],
        'tích cực': ['tích cưc','tích cục','tích cuc','tich cực','tich cưc','tich cục','tich cuc'],
        'tiêu cực': ['tiêu cưc','tiêu cục','tiêu cuc','tieu cực','tieu cưc','tieu cục','tieu cuc'],
        'hiệu quả': ['hiệu qua','hiêu quả','hiêu qua','hieu quả','hieu qua'],
        'khô khan': ['khô khán','khô khàn','khô khản','khô khạn'],        
        'cụ thể': ['cụ thê','cụ thẻ','cụ the','cu thể','cu thê','cu thẻ','cu the'],
        'lôi cuốn': ['lôi cuôn','lôi cuón','loi cuốn','loi cuôn','loi cuon'],
        'hỗ trợ': ['hồ trợ','hỗ trơ','hỗ tro','hổ trợ','hô trợ','hô trơ','hô tro','ho trợ','ho trơ','ho tro'],
        'soạn thảo': ['soản thảo','soạn thao','soan thảo','soan thao'],
        'theo dõi': ['theo giỏi','theo giõi','theo gioi','theo doi','theo dỏi'],
        'tự luận': ['tự luân','tự luạn','tự luan','tư luận','tư luân','tư luan','tu luận','tu luân','tu luận'],
        'đôi lúc': ['đội lúc','đội luc','đôi luc','đoi lúc','đoi luc','doi lúc','doi luc'],
        'đôi khi': ['đoi khi','doi khi'],
        'cập nhật': ['cập nhât','cập nhạt','cập nhat','câp nhật','câp nhạt','câp nhat','cap nhật','cap nhât','cap nhạt','cap nhat'],
        'tài liệu': ['tài liêu','tài liẹu','tài lieu','tai liệu','tai liêu','tai liẹu','tai lieu'],
        'ý kiến': ['y kiến','y kiên','y kien','ý kiên','ý kien'],
        'so với': ['so vời','so vơi','so voi','so vói','so vòi'],        
        'cần dành': ['cần giành','can danh'],
        'hòa hợp': ['hóa hợp','hòa hơp','hòa hop','hoa hợp','hoa hơp','hoa hop'],
        'phương pháp':['phương phap','phuong pháp','phuong phap'],
        'bổ ích': ['bổ ich','bổ ít','bô ích','bô ich','bo ích','bo ich'],
        ' dạy chán ': [' dạy chan ',' day chán ',' day chan ',' dạy cháng ',' dạy chang ',' day cháng '],
        'môn học': ['môn hoc','môn hóc','mon học','mon hóc','mon hoc'],
        ' tuyệt vời ': [' tuyệt vơi ',' tuyệt vòi ',' tuyệt voi ',' tuyêt vời ',' tuyêt vơi ',' tuyêt vòi ',' tuyêt voi ',' tuyet vời ',' tuyet vơi ',' tuyet voi ',' tuỵt vời ',' tuỵt vơi ',' tuỵt vòi ',' tuỵt voi ',' tuyt vời ',' tuyt vơi ',' tuyt vòi ',' tuyệt dời ',' tuyệt dơi ',' tuyet doi '],
        ' thời gian ': [' thời giang ',' thơi gian ',' thoi gian '],
        'tuy nhiên': ['tuy nhien','tuy nhiến','tuy nhiền','tuy nhiển','tuy nhiện','tuy nhién','tuy nhièn','tuy nhiẻn','tuy nhiẽn','tuy nhiẹn'],
        'cung cấp': ['cung câp','cung cap','cung cắp','cung căp','cung cáp'],
        'phòng học': ['phòng hoc','phòng hóc','phong học','phong hoc','phong hóc'],
        'xem xét': ['xem xet','xem set','xem sét','sem xét','sem xet','sem set','sem sét'],
        'vui tính': ['vui tình','vui tỉnh','vui tĩnh','vui tịnh','vui tinh'],
        'khả năng': ['khả nang','khả nâng','kha nâng','kha năng','kha nang','khã năng','khã nâng','khã nang'],
        'chu đáo': ['chu đao','chu đạo','chu đảo','chu đào','chu đão','chu dao'],
        'đề nghị': ['đề nghi','đê nghị','đê nghi','đe nghị','đe nghi','de nghị','de nghi'],
        'dạy ổn': ['dạy ôn','dạy on','day ổn','day ôn','day on'],
        ' lan man ': [' lan mang ',' lang man ',' lang mang '],
        'lớp học': ['lớp hoc','lớp hóc','lơp học','lơp hóc','lơp hoc','lop hóc','lop học','lop hoc'],
        'bài thi': ['bai thi','bái thi','bải thi','bãi thi','bại thi'],
        'bài học': ['bài hoc','bài hóc','bai học','bai hóc','bai hoc'],
        'làm biếng': ['làm biêng','làm bieng','lam biếng','lam biêng','lam bieng'],
        'cực kì': ['cực ki','cưc kì','cưc ki','cuc kì','cuc ki','cực ky','cưc kỳ','cưc ky','cuc kỳ','cuc ky'],
        'thân thiện': ['thân thiên','thân thiẹn','thân thien','than thiện','than thiên','than thiẹn','than thien'],
        'diễn đạt': ['diễn đat','diễn dạt','diễn dat','diên đạt','diên đat','diên dạt','diên dat','dien đạt','dien đat','dien dạt','dien dat'],
        'gì đâu': ['gì đau','gì dau','gi đâu','gi đau','gi dau','gí đâu','gí đau','gí dau','j đâu','j đau','j dau'],
        'thông báo': ['thông bao','thong báo','thong bao','thông bào','thông bảo','thông bão','thong bào','thong bảo','thong bão'],
        'thích thú': ['thích thu','thích thù','thích thủ','thích thũ','thích thụ','thich thu','thich thù','thich thủ','thich thũ','thich thụ'],
        'xứng đáng': ['xứng đang','xứng dáng','xứng dang','xưng đáng','xưng đang','xưng dáng','xưng dang','xung đáng','xung đang','xung dáng','xung dang'],
        'đa dạng': ['đa dang','da dạng','da dang','đa dảng','đa dãng','da dảng','da dãng'],
        'liên hệ': ['liên hê','liên hẹ','liên he','lien hệ','lien hê','lien hẹ','lien he'],
        'lý thuyết': ['lý thuyêt','lý thuýet','lý thuyet','ly thuyết','ly thuyêt','ly thuýet','ly thuyet','lí thuyêt','lí thuýet','lí thuyet','li thuyết','li thuyêt','li thuýet','li thuyet'],
        'vô cùng': ['vô cung','vo cùng','vo cung'],
        'ý nghĩa': ['ý nghia','ý nghía','ý nghìa','ý nghỉa','ý nghịa','y nghĩa','y nghia','y nghía','y nghìa','y nghỉa','y nghịa'],
        'cơ hội': ['cơ hôi','cơ hối','cơ hồi','cơ hổi','cơ hỗi','co hội','co hôi','co hối','co hồi','co hổi','co hỗi','cơ hoi','cơ họi','co hoi'],
        'thú vị': ['thú vi','thu vị','thu vi'],
        'thuận tiện': ['thuận tiên','thuận tien','thuân tiện','thuân tiên','thuân tien','thuan tiện','thuan tiên','thuan tien'],
        'trình bày': ['trình bay','trình báy','trình bảy','trình bãy','trình bạy','trinh bày','trinh báy','trinh bảy','trinh bãy','trinh bạy','trinh bay'],
        ' vận dụng ': [' vận dụn ',' vận dung ',' vân dụng ',' vân dung ',' van dụng ',' van dung '],
        'hạn hẹp': ['hạn hep','han hẹp','han hep'],
        'tương tác': ['tương tac','tuong tác','tuong tac'],
        'buồn ngủ': ['buồn ngu','buồn ngũ','buồn ngụ','buồn ngú','buôn ngủ','buôn ngu','buôn ngũ','buôn ngụ','buôn ngú','buon ngủ','buon ngu','buon ngũ','buon ngụ','buon ngú'],
        'trao đổi': ['trao đôi','trao đỏi','trao đoi','trao đỗi','trao dỏi','trao dõi','trao doi'],
        'kĩ năng': ['kĩ nang','ki năng','ki nang','kỉ năng','kỉ nang','kỹ nang','ky năng','ky nang','kỷ năng','kỷ nang'],
        'trình độ': ['trình đô','trình đọ','trình đo','trình do','trinh độ','trinh đô','trinh đọ','trinh đo','trinh do'],
        'sư phạm': ['sư pham','su phạm','su pham'],
        ' hăng say ': [' hăn say ',' hang say '],
        'câu hỏi': ['câu hõi','câu hoi','câu họi','cau hỏi','cau hõi','cau hoi','cau họi'],
        ' thời khóa biểu ': [' thời khóa biễu ',' thơi khoa biêu ',' thoi khoa bieu ',' tkb '],
        ' ứng dụng ': [' ứng dụn ',' ứn dụng ',' ứn dụn ',' ứng dung ',' ưng dụng ',' ưng dung ',' ung dụng '],
        'tương đối': ['tương đôi','tương đói','tương doi','tuong đối','tuong đôi','tuong đói','tuong doi'],
        'cải thiện': ['cải thiên','cải thiẹn','cải thien','cãi thiện','cãi thiên','cãi thiẹn','cãi thien','cai thiện','cai thiên','cai thiẹn','cai thien'],
        'tâm lý': ['tâm ly','tam lý','tam ly','tâm li','tam lí','tam li'],
        'chương trình': ['chương trinh','chuong trình','chuong trinh'],

        #-----------------------------------------------------------------------------------------------------#

        # Từ đơn:
                      
        ' về ': [' vê ',' dề ',' dềe ',' vềe ',' vềee ',' vè ',' vèe '],
        ' ngành ': [' nganh ',' ngánh ',' ngảnh ',' ngãnh ',' ngạnh ',' nghành '],
        ' được ': [' đươc ',' dc ',' dcc ',' dccc ',' đc ',' đcc ',' đccc ',' duoc ',' đuoc '],
        ' chỗ ': [' chổ ',' chỗo '],
        ' nè ': [' nà ',' nèe ',' nèee ',' nèeee '],
        ' dễ ': [' de2 ',' dễe ',' dể ',' deef '],
        ' có ' : [' co ',' cóo ',' cóoo ',' cóooo ', ' cóa ',' cóaa ',' cóaaa ', ' cóaaaa ',' coa '],
        ' giảng ' : [' giảnh ',' giang ',' giãng '],
        ' giờ ': [' giời ',' gờ ',' h ',' dờ '],
        ' dạy ': [' giạy ',' dậy học ',' day '],
        ' thêm ': [' them '],
        ' bằng ': [' bắng ',' bang '],
        ' gì ': [' gi ',' gí ',' j ',' jj ',' jjj '],
        ' không ': [' k ',' ko ',' koo ',' kooo ',' khom ',' khomm ',' khum ',' khumm ',' hong ',' hongg ',' honggg ',' hông ',' hôngg ',' hônggg ',' khong ',' khongg ',' khonggg '],
        ' rất ': [' rắt ',' rât ',' rat '],
        ' sao ': [' saoooo ',' xao ',' saoo ',' saooo ',' s '],
        ' vậy ': [' z ',' zz ',' dãy ',' dãyy ',' dãyyy ',' d ',' vay ',' vayy ',' vayyy ',' zay ',' zayy ',' zayyy ',' v '],
        ' rõ ': [' rỏ ',' rõo ',' rõoo '],
        ' dẫn ': [' dẩn ',' dẩnn ',' dẩnnn ',' dẫnn ',' dẫnnn '],
        ' nhiều ': [' nhìu ',' nhìuu ',' nhìuuu ',' nhiềuu ',' nhiềuuu '],
        ' cũng ': [' cũn ',' cũm ',' cx ',' cũngg ',' cũnggg '],
        ' luôn ': [' luon ',' luônn ',' luônnn '],
        ' buổi ': [' buôi ',' buối ',' buội ',' buỗi ',' buoi ',' bủi ',' bũi '],
        ' tốt ': [' tôt ',' tôtt ',' tot ',' tott ',' tốtt ',' tốttt '],
        ' đỉnh ': [' đink ',' đinkk ',' đĩnh ',' đĩnhh ',' dink ',' dinkk ',' đỉnhh ',' đỉnhhh '],
        ' hay ': [' hayy ',' hayyy ',' hayyyy '],
        ' nhưng ': [' nhưn ',' nhung '],
        ' biết ': [' biêt ',' biêtt ',' biêttt ',' bik ',' bikk ',' bikkk ',' bit ',' bít ',' bítt ',' bíttt ',' biếtt ',' biếttt ',' pit ',' pít ',' pítt ',' píttt '],
        ' rồi ': [' rùi ',' rùii ',' ròi ',' ròii ',' rầu ',' rầuu ',' gòi ',' gòii ',' rồii ',' rồiii ',' r '],
        ' dở ': [' dởo ',' dỏm ',' dỏmm ',' dõm ',' dõmm ',' do '],
        ' tệ ': [' cùi bắp ',' cùi mía ',' cùi ',],
        ' cách ': [' cạch ',' cach '],
        ' đúng ': [' đung ',' đún ',' đúm ',' đúngg ',' đúnggg '],
        ' thôi ': [' thoi ',' thoii ',' thoiii ',' thoiiii ',' thoai ',' thoaii ',' thoaiii ',' thoaiiii ',' hoi ',' hoii ',' hoiii ',' hoiiii ',' thui ',' thuii ',' thuiii ',' thuiiii ',' thâu ',' thâuu ',' thâuuu ',' thâuuuu ',' thou ',' thouu ',' thouuu ',' thouuuu '],
        ' ngủ ': [' ngu ',' ngu3 ',' ngủu ',' ngủuu '],
        ' tuyệt ': [' tuyệt ',' tuỵt ',' tuyệtt ',' tuyệttt ',' tuyet '],
        ' với ': [' vs ',' vớii ',' vớiii ',' dới ',' dớii ',' dớiii '],
        ' hoài ': [' quài ',' quàii ',' quàiii ',' hoàii ',' hoàiii '],
        ' trời ': [' xời ',' xờii ',' xờiii ',' trờii ',' trờiii ',' troi ',' tr '],
        ' trường ': [' trương ',' truong '],
        ' lớp ': [' lop ',' lơp '],
        ' như ': [' nhu ',' nhưu '],
        ' giỏi ': [' gioi ',' giói ',' giòi ',' giõi ',' giọi ',' dỏi '],
        ' cao ': [' kao ',' caoo ',' kaoo '],
        ' hết ': [' het ',' hêt ',' hếtt '],
        ' nữa ': [' nưa ',' nựa ',' nx ',' nữaa ',' nữaaa '],
        ' đi ': [' di '],
        ' điểm ': [' điêm ',' điem ',' diem '],
        ' giữa ': [' giưa ', ' giua '],
        ' giúp ': [' giup ',' giụp '],
        ' thầy ': [' thày ']
        }

translate_dict = {
        ' bài trình chiếu ': [' slide ','slides'],
        ' đầy đủ ': [' full '],
        ' hội thảo ': [' seminar '],
        ' thời hạn ': [' deadline ','deadlines', ' dealine ',' dl ',' dline '],
        ' kiểm tra ': [' test ','tests',' check ','checks'],
        ' tổng quan ': [' overview '],
        ' được ': [' ok ',' oke ',' okela ',' okelaa ','okelaaa',' okay ',' okee ','okeee',' oki ',' okii ','okiii'],
        ' bộ dụng cụ ': [' kit '],
        ' thực hành ': [' lab '],
        ' đăng tải ' : [' post ',' up '],
        ' tải lên ' : [' upload ','uploads'],
        ' chủ đề ' :[' topic '],
        ' bài học ' : [' unit ','units'],
        ' trò chơi ' :[' game ','games'],
        ' phần mềm ' : [' wireshark ', ' silverlight ',' proteus '],
        ' sao chép ' :[' copy ',' copies ','copys'],
        ' dán ' : [' paste ','pastes'],
        ' tệp ' : [' file ','files'],
        ' giao diện ' : [' console ',' win form '],
        ' mô hình ứng dụng ' : [' pattern serverside ',' lập tình windows ',' lập trình window ',' lập trình windows '],
        ' lập trình hướng đối tượng ' : [' oop ',' java '], 
        ' ngôn ngữ lập trình ' : [' prolog '],
        ' trang mạng ' :[' web ',' webs ',' wed ', ' website ','websites',' progressive web app ',' web app ',' web apps ',' webs app ',' webs apps ',' amp ',' seo '],
        ' sách điện tử ': [' ebook ','ebooks', ' e-book ','e-books'],
        ' công ty ' : [' altera '],
        ' ngôn ngữ truy vấn ' : [' sql '],
        ' dự án ': [' project ',' proj '],
        ' chia sẻ ': [' share '],
        ' cập nhật ': [' update '],
        ' công nghệ thông tin ': [' it '],
        ' trả lời ': ['reply',' rep '],
        ' không ': [' nope ',' no ',' not '],
        ' tiếng anh ': [' english ',' engrisk ',' englisk ',' eng '],
        ' là ': [ ' is ',' are ',' was ',' were '],
        ' rất ': [' very '],
        ' tốt ': [' good ',' gud ',' gút ',' gut ',' gudd ',' best ',' the best ',' bestt ','besttt'],
        ' khóa học ': [' courses ',' crs ',' course ',' coursera ','courseras'],
        ' trình bày thử ': [' demo '],
        ' hoàn hảo ': [' perfect ',' perf ','perfection'],
        ' nhắn tin ': [' chat ',' chats ',' ib ',' inbox ',' ibox ',' chitchat ',' chatting '],
        ' tư duy ': [' logic ','logics'],
        ' nộp ': [' submit '],
        ' trẻ trung ': [' teen '],
        ' ngoài trời ' : [' outdoor ','outdoors'],
        ' hoạt động ': [' activity ', ' activities '],
        ' vui ' :[' funny ', ' fun ','funnys','funnies'],
        ' hệ điều hành ' : [' linux ',' windows ', ' win server '],
        ' góp ý ' : [' feed back ',' feed backs ',' feedback ',' feedbacks '],
        ' phương pháp học tập ' : [' projectbase ','projectbases',' project-base ','project-bases'],
        ' nhóm ' : [' team ',' group '],
        ' dễ thương ' : [' cute ',' cutee ','cuteee'],
        ' đào tạo ' : [' train ','trains',' training ','trainings','trained'],
        ' thích ' : [' like ','likes'],
        ' thất bại ' : ['failse',' fail ','failed',' fails '],
        ' ông chủ ' : [' boss ','bosses'],
        ' phòng trò chuyện ' : [' chat room ','chat rooms'],
        ' thêm ' : [' add ','addition'],
        ' tên ' : [' name ','names'],
        ' kích cỡ ' : [' size ','sizes'],
        ' thiết kế phần mềm ' : [' design pattern ',' design patterns ',' designs pattern ',' designs patterns '],
        ' sơ đồ luồng dữ liệu ' : [' dfd '],
        ' phiên bản ' : [' version ',' versions '],
        ' vi mạch tích hợp ' : [' nfc ',' NearField Communication ',' NearField Communications ',' Near-Field Communication ', ' Near-Field Communications '],
        ' dễ ' : [' easy ',' ez ',' easies ',' easys '],
        'từng bước một' : ['step by step','steps by steps','steps by step','step by steps'],
        ' quá giờ ': [' over time ',' over times ',' overtime ',' overtimes '],
        ' buổi thuyết trình ': [' buổi present ',' buổi presents ',' presentation ',' presentations '],
        ' thuyết trình ': [' present ',' presents '],
        ' tải ': [' down ',' downs ',' download ',' downloads '],
        ' phong cách ': [' style ',' styles '],
        ' cực kì ': [' max ',' maxx ',' maxxx '],
        ' cấp độ ': [' level ',' levels ',' lv '],
        ' bình luận ': [' comment ',' comments ',' cmt ',' cmts '],
        ' sách giáo khoa ': [' textbook ',' textbooks ',' book ',' books '],
        ' cơ bản ': [' basic ',' basics '],
        ' phản biện ': [' debate ',' debates '],
        ' trả lời tin nhắn ': [' rep mail ',' rep mails ',' reply mail ',' reply mails '],
        ' nhắn tin ': [' mail ',' mails '],
        ' đỉnh ': [' pro ',' proo ',' prooo ',' vip ',' vipp ',' vippp '],
        ' bỏ phiếu ': [' vote ',' votes ']
        }



number_dict = {
    '0%': 'không phần trăm',
    '5%': 'năm phần trăm',
    '10%': 'mười phần trăm',
    '15%': 'mười lăm phần trăm',
    '20%': 'hai mươi phần trăm',
    '25%': 'hai mươi lăm phần trăm',
    '30%': 'ba mươi phần trăm',
    '35%': 'ba mươi lăm phần trăm',
    '40%': 'bốn mươi phần trăm',
    '45%': 'bốn mươi lăm phần trăm',
    '50%': 'năm mươi phần trăm',
    '55%': 'năm mươi lăm phần trăm',
    '60%': 'sáu mươi phần trăm',
    '65%': 'sáu mươi lăm phần trăm',
    '70%': 'bảy mươi phần trăm',
    '75%': 'bảy mươi lăm phần trăm',
    '80%': 'tám mươi phần trăm',
    '85%': 'tám mươi lăm phần trăm',
    '90%': 'chín mươi phần trăm',
    '95%': 'chín mươi lăm phần trăm',
    '100%': 'trăm phần trăm',



    '0': 'không',
    '1': 'một',
    '2': 'hai',
    '3': 'ba',
    '4': 'bốn',
    '5': 'năm',
    '6': 'sáu',
    '7': 'bảy',
    '8': 'tám',
    '9': 'chín',
    '10': 'mười',
    '11': 'mười một',
    '12': 'mười hai',
    '13': 'mười ba',
    '14': 'mười bốn',
    '15': 'mười lăm',
    '16': 'mười sáu',
    '17': 'mười bảy',
    '18': 'mười tám',
    '19': 'mười chín',
    '20': 'hai mươi',
    '21': 'hai mươi mốt',
    '22': 'hai mươi hai',
    '23': 'hai mươi ba',
    '24': 'hai mươi bốn',
    '25': 'hai mươi lăm',
    '26': 'hai mươi sáu',
    '27': 'hai mươi bảy',
    '28': 'hai mươi tám',
    '29': 'hai mươi chín',
    '30': 'ba mươi',
    '31': 'ba mươi mốt',
    '32': 'ba mươi hai',
    '33': 'ba mươi ba',
    '34': 'ba mươi bốn',
    '35': 'ba mươi lăm',
    '36': 'ba mươi sáu',
    '37': 'ba mươi bảy',
    '38': 'ba mươi tám',
    '39': 'ba mươi chín',
    '40': 'bốn mươi',
    '41': 'bốn mươi mốt',
    '42': 'bốn mươi hai',
    '43': 'bốn mươi ba',
    '44': 'bốn mươi bốn',
    '45': 'bốn mươi lăm',
    '46': 'bốn mươi sáu',
    '47': 'bốn mươi bảy',
    '48': 'bốn mươi tám',
    '49': 'bốn mươi chín',
    '50': 'năm mươi',
    '51': 'năm mươi mốt',
    '52': 'năm mươi hai',
    '53': 'năm mươi ba',
    '54': 'năm mươi bốn',
    '55': 'năm mươi lăm',
    '56': 'năm mươi sáu',
    '57': 'năm mươi bảy',
    '58': 'năm mươi tám',
    '59': 'năm mươi chín',
    '60': 'sáu mươi',
    '61': 'sáu mươi mốt',
    '62': 'sáu mươi hai',
    '63': 'sáu mươi ba',
    '64': 'sáu mươi bốn',
    '65': 'sáu mươi lăm',
    '66': 'sáu mươi sáu',
    '67': 'sáu mươi bảy',
    '68': 'sáu mươi tám',
    '69': 'sáu mươi chín',
    '70': 'bảy mươi',
    '71': 'bảy mươi mốt',
    '72': 'bảy mươi hai',
    '73': 'bảy mươi ba',
    '74': 'bảy mươi bốn',
    '75': 'bảy mươi lăm',
    '76': 'bảy mươi sáu',
    '77': 'bảy mươi bảy',
    '78': 'bảy mươi tám',
    '79': 'bảy mươi chín',
    '80': 'tám mươi',
    '81': 'tám mươi mốt',
    '82': 'tám mươi hai',
    '83': 'tám mươi ba',
    '84': 'tám mươi bốn',
    '85': 'tám mươi lăm',
    '86': 'tám mươi sáu',
    '87': 'tám mươi bảy',
    '88': 'tám mươi tám',
    '89': 'tám mươi chín',
    '90': 'chín mươi',
    '91': 'chín mươi mốt',
    '92': 'chín mươi hai',
    '93': 'chín mươi ba',
    '94': 'chín mươi bốn',
    '95': 'chín mươi lăm',
    '96': 'chín mươi sáu',
    '97': 'chín mươi bảy',
    '98': 'chín mươi tám',
    '99': 'chín mươi chín',
    '100': 'một trăm'
    }