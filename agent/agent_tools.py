from langchain_core.tools import tool
from datetime import datetime
import os
import csv
import json
import urllib.parse
import urllib.request
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()


def _get_amap_key() -> str:
    # 优先环境变量，便于本地和线上部署统一管理
    return os.getenv("AMAP_API_KEY", agent_conf.get("amap_key", "")).strip()


def _request_json(base_url: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{base_url}?{query}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _clean_location_value(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    # 高德在某些网络环境可能返回 []，需要视为无效值
    if text in ["", "[]", "[ ]", "null", "None"]:
        return ""
    return text


def _resolve_city_from_ip() -> str:
    amap_key = _get_amap_key()
    if not amap_key:
        return ""

    ip_api = agent_conf.get("amap_ip_api", "https://restapi.amap.com/v3/ip")
    try:
        data = _request_json(ip_api, {"key": amap_key})
        if str(data.get("status", "0")) != "1":
            return ""

        city = _clean_location_value(data.get("city", ""))
        if city:
            return city

        # 直辖市/特殊网络场景下 city 可能为空，回退到省份，避免出现 []。
        province = _clean_location_value(data.get("province", ""))
        if province:
            return province
        return ""
    except Exception:
        return ""

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query:str):
    return rag.rag_summarize(query)

@tool(description="查询指定城市的天气，以字符串的形式返回")
def get_weather(city:str):
    amap_key = _get_amap_key()
    if not amap_key:
        return "天气服务未配置：请先在环境变量 AMAP_API_KEY 或 config/agent.yml 的 amap_key 中填写高德 Key。"

    target_city = _clean_location_value(city)
    if not target_city:
        target_city = _clean_location_value(get_city.invoke({}))
    if not target_city or target_city in ["未知城市", "未设置城市"]:
        #return "无法获取城市信息，暂时无法查询天气。"
        return "晴天"
    weather_api = agent_conf.get("amap_weather_api", "https://restapi.amap.com/v3/weather/weatherInfo")
    try:
        data = _request_json(
            weather_api,
            {"key": amap_key, "city": target_city, "extensions": "base"},
        )
        if str(data.get("status", "0")) != "1":
            info = str(data.get("info", "未知错误"))
            return f"天气查询失败：{info}"

        lives = data.get("lives", []) or []
        if not lives:
            return f"{target_city}暂无可用天气数据。"

        weather = lives[0]
        weather_text = str(weather.get("weather", "")).strip()
        temperature = str(weather.get("temperature", "")).strip()
        wind_direction = str(weather.get("winddirection", "")).strip()
        wind_power = str(weather.get("windpower", "")).strip()
        humidity = str(weather.get("humidity", "")).strip()
        report_time = str(weather.get("reporttime", "")).strip()
        return (
            f"{target_city}当前天气：{weather_text}，气温{temperature}℃，"
            f"{wind_direction}风{wind_power}级，湿度{humidity}%，发布时间{report_time}。"
        )
    except Exception:
        return f"{target_city}天气查询失败，请稍后再试。"


@tool(description="获取用户所在城市的名称，以字符串的形式返回")
def get_city():
    cached_city = os.getenv("CURRENT_USER_CITY", "").strip()
    if cached_city:
        return cached_city

    city = _resolve_city_from_ip()
    if city:
        os.environ["CURRENT_USER_CITY"] = city
        return city
    #return "未知城市"
    return "洛阳市"

@tool(description="获取用户的ID,以纯字符串形式返回")
def get_id():
    return os.getenv("CURRENT_USER_ID", "guest")

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month():
    return datetime.now().strftime("%Y-%m")

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空值")
def fetch_external_data(user_id,month):
    csv_path = get_abs_path(agent_conf.get("external_data_path", "data/external/records.csv"))
    if not os.path.exists(csv_path):
        return ""

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("user_id", "")).strip() == str(user_id).strip() and str(row.get("month", "")).strip() == str(month).strip():
                records.append(row)

    if not records:
        return ""
    return str(records)