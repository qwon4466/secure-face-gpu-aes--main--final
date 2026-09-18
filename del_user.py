import sqlite3
import os

DB_PATH = "security_system.db"

def delete_user_by_name(target_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        search_pattern = f"{target_name}%" 
        cursor.execute("SELECT name, image_path FROM users WHERE name LIKE ?", (search_pattern,))
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            # 서버가 알 수 있도록 실패(False)와 메시지를 반환합니다.
            return False, f"DB에 '{target_name}' 이(가) 포함된 데이터가 없습니다."

        # 이미지 파일 삭제
        for row in rows:
            file_path = row[1]
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

        # DB 기록 삭제
        cursor.execute("DELETE FROM users WHERE name LIKE ?", (search_pattern,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        # 서버가 알 수 있도록 성공(True)과 메시지를 반환합니다.
        return True, f"'{target_name}' 님의 모든 데이터({deleted_count}건)가 완벽하게 삭제되었습니다."

    except Exception as e:
        return False, f"에러 발생: {e}"

# ⭐️ 핵심 변경: 이 파일만 단독으로 실행할 때만 아래 코드가 작동합니다.
# (웹 서버에서 import 할 때는 실행되지 않음)
if __name__ == "__main__":
    delete_user_by_name("테스트이름")