# %% [markdown]
# # 🚀 Chatbot Tư vấn Khóa học trên Kaggle
# - Sử dụng Llama 3 qua Groq API
# - Tìm kiếm dữ liệu từ file CSV
# - Hỗ trợ function calling

# %% [code]
# Cài đặt thư viện cần thiết


# %% [code]
import pandas as pd
import json
from groq import Groq

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# %% [markdown]
# ## 1. Chuẩn bị Dữ liệu Khóa học

# %% [code]
# Tạo dữ liệu mẫu (hoặc upload file CSV của bạn lên Kaggle)
data = {
    "ten_khoa_hoc": [
        "Data Warehouse",
        "Power BI"
    ],
    "mo_ta": [
        "Xây dựng kho dữ liệu, ETL, Datamart, báo cáo BI",
        "Phân tích dữ liệu và tạo báo cáo trực quan bằng Power BI"
    ],
    "muc_tieu": [
        "Hiểu kiến trúc DWH, biết thiết kế ETL pipeline, tạo báo cáo từ dữ liệu thô",
        "Thành thạo tạo dashboard, visual hóa và trình bày dữ liệu trên Power BI"],
    "noi_dung": [
        [
            "Tổng quan về hệ thống Data Warehouse",
            "Thiết kế và xây dựng ETL pipeline",
            "Tạo Datamart và trình bày dữ liệu",
            "Tích hợp và báo cáo với công cụ BI"
        ],
        [
            "Giới thiệu Power BI và mô hình dữ liệu",
            "Kết nối, làm sạch và biến đổi dữ liệu",
            "Tạo visual (biểu đồ, bảng, KPI)",
            "Thiết kế dashboard và xuất báo cáo"
        ]
    ],
    "doi_tuong": [
        "Người muốn theo nghề Data Engineer, BI Developer, hoặc hiểu sâu về hệ thống dữ liệu",
        "Người làm phân tích dữ liệu, nhân viên văn phòng, kế toán, hoặc người mới học phân tích"
    ],
    "hoc_phi": [
        10000000,
        3500000
    ],
    "danh_muc": [
        "Data Engineer",
        "Phân tích"
    ]
}

df = pd.DataFrame(data)
df.to_csv("khoa_hoc.csv", index=False)
print("✅ Đã tạo file khoa_hoc.csv")

# Đọc lại dữ liệu để đảm bảo consistency
df = pd.read_csv("khoa_hoc.csv")
available_courses = df.to_dict(orient='records')

# %% [markdown]
# ## 2. Cấu hình Groq & Llama 3

# %% [code]
# Lấy API key từ Kaggle Secrets
client = Groq(api_key="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")


# %% [markdown]
# ## 3. Định nghĩa Hàm Tìm kiếm

def submit_order_to_sheet(ho_ten, so_dien_thoai, dia_chi, facebook, don_hang):
    try:
        # Kiểm tra thông tin bắt buộc
        if not all([ho_ten, so_dien_thoai, don_hang]):
            return "❌ Vui lòng nhập đầy đủ họ tên và số điện thoại"

        # Kết nối Google Sheet
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",  # Quyền chỉnh sửa Sheet
            "https://www.googleapis.com/auth/drive"  # Quyền truy cập Drive (nếu cần)
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        # Ghi dữ liệu
        sheet = client.open("DanhSachDonHang").sheet1
        sheet.append_row([
            ho_ten,
            so_dien_thoai,
            dia_chi or "Không có",
            facebook or "Không có",
            don_hang[0]["san_pham"],
            don_hang[0]["so_luong"],
            don_hang[0]["don_gia"],

        ])

        return "✅ Đăng ký thành công! Chúng tôi sẽ liên hệ bạn sớm."

    except Exception as e:
        return f"❌ Lỗi hệ thống: {str(e)}"
# %% [code]
def search_courses(query: str, max_results: int = 3, danh_muc: str = None) -> list:
    # Đảm bảo max_results luôn là số nguyên dương
    try:
        max_results = int(max_results)
        if max_results <= 0:
            max_results = 3
    except (ValueError, TypeError):
        max_results = 3  # Giá trị mặc định nếu chuyển đổi thất bại

    query = query.lower()

    # Tìm kiếm các khóa học phù hợp với từ khóa
    results = df[
        df.apply(lambda row: any(
            query in str(value).lower()
            for value in row.values
        ), axis=1)
    ]

    # Lọc thêm theo danh mục nếu có
    if danh_muc:
        results = results[results["danh_muc"].str.lower() == danh_muc.lower()]

    # Xử lý kết quả
    if results.empty:
        return [{"message": "Không tìm thấy khóa học nào phù hợp với yêu cầu."}]

    # Giới hạn số lượng kết quả
    n_results = min(len(results), max_results)

    # Trả về danh sách khóa học
    return results.head(n_results)[[
        "ten_khoa_hoc", "mo_ta", "hoc_phi", "danh_muc", "noi_dung", "doi_tuong", "muc_tieu"
    ]].to_dict(orient="records")


tools = [{
    "type": "function",
    "function": {
        "name": "search_courses",
        "description": "Tìm kiếm khóa học phù hợp từ cơ sở dữ liệu",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Từ khóa tìm kiếm (tên khóa học, mô tả, danh mục)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Số lượng kết quả tối đa mà hệ thống sẽ trả về (mặc định: 3)",
                    "default": 3
                },
                "danh_muc": {
                    "type": "string",
                    "description": "Danh mục để lọc các khóa học (ví dụ: 'Lập trình', 'AI'). Nếu không cung cấp, hệ thống sẽ tìm kiếm trong tất cả danh mục."
                }
            },
            "required": ["query"]
        },
        "returns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ten_khoa_hoc": {
                        "type": "string",
                        "description": "Tên của khóa học"
                    },
                    "mo_ta": {
                        "type": "string",
                        "description": "Mô tả ngắn gọn về khóa học"
                    },
                    "hoc_phi": {
                        "type": "integer",
                        "description": "Học phí của khóa học"
                    },
                    "danh_muc": {
                        "type": "string",
                        "description": "Danh mục mà khóa học thuộc về (ví dụ: 'Lập trình', 'AI')"
                    },
                    "noi_dung": {
                        "type": "string",
                        "description": "Nội dung chi tiết của khóa học"
                    },
                    "doi_tuong": {
                        "type": "string",
                        "description": "Đối tượng phù hợp với khóa học"
                    },
                    "muc_tieu": {
                        "type": "string",
                        "description": "Mục tiêu mà người học sẽ đạt được khi hoàn thành khóa học"
                    }
                },
                "required": ["ten_khoa_hoc", "mo_ta", "hoc_phi", "danh_muc"]
            }
        }
    }
}

    ,
    {
        "type": "function",
        "function": {
            "name": "submit_order_to_sheet",
            "description": "khác hàng muốn đăng kí khóa học",
            "parameters": {
                "type": "object",
                "properties": {
                    "ho_ten": {
                        "type": "string",
                        "description": "Họ và tên người mua"
                    },
                    "so_dien_thoai": {
                        "type": "string",
                        "description": "Số điện thoại liên hệ"
                    },
                    "dia_chi": {
                        "type": "string",
                        "description": "Địa chỉ giao hàng hoặc thanh toán"
                    },
                    "facebook": {
                        "type": "string",
                        "description": "Link Facebook cá nhân của người mua"
                    },
                    "don_hang": {
                        "type": "array",
                        "description": "Danh sách sản phẩm trong đơn hàng",
                        "items": {
                            "type": "object",
                            "properties": {
                                "san_pham": {
                                    "type": "string",
                                    "description": "Tên sản phẩm hoặc khóa học"
                                },
                                "so_luong": {
                                    "type": "integer",
                                    "description": "Số lượng sản phẩm"
                                },
                                "don_gia": {
                                    "type": "number",
                                    "description": "Đơn giá của sản phẩm"
                                },
                                "thanh_tien": {
                                    "type": "number",
                                    "description": "Thành tiền = đơn giá * số lượng"
                                }
                            },
                            "required": ["san_pham", "so_luong", "don_gia", "thanh_tien"]
                        }
                    }
                },
                "required": ["ho_ten", "so_dien_thoai", "don_hang"]
            }
        }
    }]


# %% [markdown]
# ## 5. Hàm Hỏi Đáp với Llama 3


def ask_llama(full_prompt: str) -> str:
    chat_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""Bạn là trợ lý tư vấn khóa học thông minh. CHỈ đề xuất các khóa học từ danh sách sau:
                {available_courses}

QUY TẮC:
1. Nếu không tìm thấy khóa học phù hợp, trả lời: 'Xin lỗi, hiện chúng tôi chưa có khóa học về chủ đề này.'
2. Luôn kiểm tra kỹ thông tin trước khi trả lời.
3. Nếu người dùng muốn đăng ký nhưng thiếu thông tin (họ tên, số điện thoại, tên khóa học), chỉ yêu cầu những thông tin còn thiếu.
4. Nếu thông tin đầy đủ, thực hiện đăng ký luôn.
"""
            },
            {"role": "user", "content": full_prompt}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.7,
        max_tokens=1024
    )

    response_message = chat_response.choices[0].message

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            func_args = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "search_courses":
                print("🔍 Đang tìm kiếm trong cơ sở dữ liệu...")
                courses = search_courses(
                    query=func_args.get("query"),
                    max_results=func_args.get("max_results", 3),
                    danh_muc=func_args.get("danh_muc")
                )

                print(courses)

                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": f"""Bạn là trợ lý tư vấn khóa học thông minh. CHỈ đề xuất các khóa học từ danh sách sau:
                {courses}
                        QUY TẮC:
1. Nếu không tìm thấy khóa học phù hợp, trả lời: 'Xin lỗi, hiện chúng tôi chưa có khóa học về chủ đề này.'
2. Luôn kiểm tra kỹ thông tin trước khi trả lời.
3. Nếu người dùng muốn đăng ký nhưng thiếu thông tin (họ tên, số điện thoại, tên khóa học), chỉ yêu cầu những thông tin còn thiếu.
4. Nếu thông tin đầy đủ, thực hiện đăng ký luôn.
"""}
                        ,
                        {"role": "user", "content": full_prompt},
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        {
                            "role": "tool",
                            "name": "search_courses",
                            "content": json.dumps(courses, ensure_ascii=False),
                            "tool_call_id": tool_call.id
                        }
                    ],
                    temperature=0.7
                )
                return final_response.choices[0].message.content

            elif tool_call.function.name == "submit_order_to_sheet":
                # Lấy các thông tin cần thiết
                missing_fields = []
                ho_ten = func_args.get("ho_ten")
                so_dien_thoai = func_args.get("so_dien_thoai")
                don_hang = func_args.get("don_hang")

                if not ho_ten:
                    missing_fields.append("họ tên")
                if not so_dien_thoai:
                    missing_fields.append("số điện thoại")
                if not don_hang:
                    missing_fields.append("tên khóa học")

                if missing_fields:
                    return f"❗Vui lòng cung cấp thêm: {', '.join(missing_fields)}."

                result = submit_order_to_sheet(
                    ho_ten=ho_ten,
                    so_dien_thoai=so_dien_thoai,
                    dia_chi=func_args.get("dia_chi", "Không có"),
                    facebook=func_args.get("facebook", "Không có"),
                    don_hang=don_hang
                )
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": f"""Bạn là trợ lý tư vấn khóa học thông minh. CHỈ đề xuất các khóa học từ danh sách sau:
                                        {result}
                                                QUY TẮC:
                        1. Nếu không tìm thấy khóa học phù hợp, trả lời: 'Xin lỗi, hiện chúng tôi chưa có khóa học về chủ đề này.'
                        2. Luôn kiểm tra kỹ thông tin trước khi trả lời.
                        3. Nếu người dùng muốn đăng ký nhưng thiếu thông tin (họ tên, số điện thoại, tên khóa học), chỉ yêu cầu những thông tin còn thiếu.
                        4. Nếu thông tin đầy đủ, thực hiện đăng ký luôn.
                        """},
                        {"role": "user", "content": full_prompt},
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        {
                            "role": "tool",
                            "name": "search_courses",
                            "content": json.dumps(result, ensure_ascii=False),
                            "tool_call_id": tool_call.id
                        }
                    ],
                    temperature=0.7
                )
                return final_response.choices[0].message.content



    return response_message.content



#while True:
#   user_input = input("Bạn: ")
#   if user_input.lower() in ['exit', 'quit', 'tạm biệt']:
#        print("Chatbot: Cảm ơn bạn! Nếu cần tư vấn thêm, hãy quay lại nhé.")
#        break
#   response = ask_llama(user_input)
#   print(f"Chatbot: {response}")