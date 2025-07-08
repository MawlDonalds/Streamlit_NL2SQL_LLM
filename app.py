import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.callbacks import BaseCallbackHandler
import re
import numpy as np

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="BI Dashboard", page_icon="📊", layout="wide")

# Kelas untuk menangkap query SQL dan log
class SQLCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.sql_query = None
        self.logs = []
        self.all_queries = []
        self.raw_data = []  # Untuk menyimpan semua data mentah

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.logs.append(f"=== TOOL START ===")
        self.logs.append(f"Serialized: {serialized}")
        self.logs.append(f"Input type: {type(input_str)}")
        self.logs.append(f"Input content: {input_str}")
        self.raw_data.append(('tool_start', serialized, input_str))
        
        # Coba tangkap query dari berbagai format input
        query_found = self._extract_query_from_data(input_str)
        if query_found:
            self.logs.append(f"✓ Query found in tool_start: {query_found}")

    def on_tool_end(self, output, **kwargs):
        self.logs.append(f"=== TOOL END ===")
        self.logs.append(f"Output type: {type(output)}")
        self.logs.append(f"Output content: {output}")
        self.raw_data.append(('tool_end', output))
        
        query_found = self._extract_query_from_data(output)
        if query_found:
            self.logs.append(f"✓ Query found in tool_end: {query_found}")

    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"=== AGENT ACTION ===")
        self.logs.append(f"Action: {action}")
        self.logs.append(f"Action type: {type(action)}")
        
        if hasattr(action, 'tool'):
            self.logs.append(f"Tool: {action.tool}")
        if hasattr(action, 'tool_input'):
            self.logs.append(f"Tool input type: {type(action.tool_input)}")
            self.logs.append(f"Tool input: {action.tool_input}")
            
            query_found = self._extract_query_from_data(action.tool_input)
            if query_found:
                self.logs.append(f"✓ Query found in agent_action: {query_found}")
        
        self.raw_data.append(('agent_action', action))

    def on_agent_end(self, output, **kwargs):
        self.logs.append(f"=== AGENT END ===")
        self.logs.append(f"Output: {output}")
        self.raw_data.append(('agent_end', output))

    def on_text(self, text, **kwargs):
        self.logs.append(f"=== TEXT CALLBACK ===")
        self.logs.append(f"Text: {text}")
        self.raw_data.append(('text', text))
        
        query_found = self._extract_query_from_data(text)
        if query_found:
            self.logs.append(f"✓ Query found in text: {query_found}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.logs.append(f"=== CHAIN START ===")
        self.logs.append(f"Serialized: {serialized}")
        self.logs.append(f"Inputs: {inputs}")
        self.raw_data.append(('chain_start', serialized, inputs))

    def on_chain_end(self, outputs, **kwargs):
        self.logs.append(f"=== CHAIN END ===")
        self.logs.append(f"Outputs: {outputs}")
        self.raw_data.append(('chain_end', outputs))

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logs.append(f"=== LLM START ===")
        self.logs.append(f"Prompts: {prompts}")
        self.raw_data.append(('llm_start', serialized, prompts))

    def on_llm_end(self, response, **kwargs):
        self.logs.append(f"=== LLM END ===")
        self.logs.append(f"Response: {response}")
        self.raw_data.append(('llm_end', response))
        
        # Coba ekstrak dari response LLM
        if hasattr(response, 'generations'):
            for gen in response.generations:
                for g in gen:
                    if hasattr(g, 'text'):
                        query_found = self._extract_query_from_data(g.text)
                        if query_found:
                            self.logs.append(f"✓ Query found in llm_end: {query_found}")

    def _extract_query_from_data(self, data):
        """Ekstrak query dari berbagai format data"""
        queries_found = []
        
        if isinstance(data, dict):
            # Cek semua key yang mungkin berisi query
            for key in ['query', 'sql', 'statement', 'command']:
                if key in data:
                    query = data[key]
                    if self._is_select_query(query):
                        queries_found.append(query)
                        self.sql_query = query
                        self.all_queries.append(query)
        
        elif isinstance(data, str):
            # Cek apakah seluruh string adalah query
            if self._is_select_query(data):
                queries_found.append(data)
                self.sql_query = data
                self.all_queries.append(data)
            
            # Coba ekstrak query dengan regex
            patterns = [
                r'```sql\s*(.*?)\s*```',
                r'```\s*(SELECT.*?)\s*```',
                r'Query:\s*(SELECT.*?)(?=\n|$)',
                r'(SELECT\s+.*?FROM\s+.*?)(?=\n\n|\n(?=[A-Z])|$)',
                r'(SELECT\s+.*?FROM\s+\w+(?:\s+WHERE\s+.*?)?(?:\s+GROUP\s+BY\s+.*?)?(?:\s+ORDER\s+BY\s+.*?)?(?:\s+LIMIT\s+\d+)?)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, data, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    match = match.strip()
                    if self._is_select_query(match):
                        queries_found.append(match)
                        self.sql_query = match
                        self.all_queries.append(match)
        
        return queries_found[0] if queries_found else None

    def _is_select_query(self, text):
        """Cek apakah text adalah query SELECT yang valid"""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return False
        
        text_upper = text.upper()
        return ("SELECT" in text_upper and 
                "FROM" in text_upper and 
                len(text.strip()) > 15)

    def get_sql_query(self):
        return self.sql_query

    def get_all_queries(self):
        return list(set(self.all_queries))  # Remove duplicates

    def get_logs(self):
        return "\n".join(self.logs)
    
    def get_raw_data(self):
        return self.raw_data

# Konfigurasi Gemini API (ganti dengan API key Anda)
GOOGLE_API_KEY = "AIzaSyC1U2KrsHXvaS1Vx73rCcR9KPn70qCzdyU"  # Ganti dengan API key Anda

# Setup koneksi ke PostgreSQL (ganti dengan kredensial Anda)
DB_USER = "postgres"
DB_PASSWORD = "admin123"  # Ganti dengan password Anda
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "sales"  # Ganti dengan nama database Anda
connection_string = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Inisialisasi database
try:
    db = SQLDatabase.from_uri(connection_string)
    st.write("✅ Berhasil terhubung ke database")
    tables = db.get_usable_table_names()
    st.write(f"📊 Tabel yang terdeteksi: {', '.join(tables)}")
except Exception as e:
    st.error(f"❌ Gagal terhubung ke database: {str(e)}")
    st.stop()

# Inisialisasi LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
except Exception as e:
    st.error(f"❌ Gagal menginisialisasi LLM: {str(e)}")
    st.stop()

# Buat SQL agent
sql_callback = SQLCaptureCallback()
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    verbose=True,
    callbacks=[sql_callback]
)

# Inisialisasi SQLAlchemy engine untuk visualisasi
engine = create_engine(connection_string)

# Fungsi untuk mendeteksi kolom numerik, kategorikal, dan tanggal
def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Cek kolom yang mungkin tanggal tapi bertipe object
    for col in categorical_cols.copy():
        if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
            try:
                pd.to_datetime(df[col].head(10))
                date_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass
    
    return numeric_cols, categorical_cols, date_cols

# Fungsi untuk membuat visualisasi berdasarkan prompt dan data
def create_smart_visualization(df, prompt):
    try:
        if df.empty:
            return None
        
        # Pastikan tipe data
        if 'sale_date' in df.columns:
            df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        charts = []
        prompt_lower = prompt.lower()
        
        # 1. Line chart untuk tren waktu (jika ada tanggal dan numerik)
        if date_cols and numeric_cols:
            try:
                for date_col in date_cols:
                    df_time = df.groupby(date_col).agg({col: 'sum' for col in numeric_cols}).reset_index()
                    for num_col in numeric_cols:
                        fig = px.line(df_time, x=date_col, y=num_col, 
                                    title=f"📈 Trend {num_col} Over Time",
                                    labels={date_col: 'Tanggal', num_col: num_col.title()})
                        fig.update_layout(xaxis_title="Tanggal", yaxis_title=num_col.title())
                        charts.append(fig)
            except Exception as e:
                st.warning(f"⚠️ Error creating line chart: {str(e)}")
                pass
        
        # 2. Bar chart untuk data kategorikal dan numerik
        if categorical_cols and numeric_cols:
            try:
                for cat_col in categorical_cols[:2]:  # Maksimal 2 kategori
                    for num_col in numeric_cols[:2]:  # Maksimal 2 numerik
                        df_cat = df.groupby(cat_col).agg({num_col: 'sum'}).reset_index()
                        df_cat = df_cat.sort_values(num_col, ascending=False).head(10)
                        fig = px.bar(df_cat, x=cat_col, y=num_col,
                                   title=f"📊 {num_col.title()} per {cat_col.title()}",
                                   labels={cat_col: cat_col.title(), num_col: num_col.title()})
                        fig.update_layout(xaxis_title=cat_col.title(), yaxis_title=num_col.title())
                        charts.append(fig)
            except Exception as e:
                st.warning(f"⚠️ Error creating bar chart: {str(e)}")
                pass
        
        # 3. Pie chart untuk distribusi kategorikal (jika ada numerik dan data tidak terlalu banyak)
        if categorical_cols and numeric_cols and len(df) <= 20:
            try:
                for cat_col in categorical_cols[:1]:  # Maksimal 1 kategori
                    for num_col in numeric_cols[:1]:  # Maksimal 1 numerik
                        df_cat = df.groupby(cat_col).agg({num_col: 'sum'}).reset_index()
                        df_cat = df_cat.sort_values(num_col, ascending=False).head(10)
                        fig = px.pie(df_cat, names=cat_col, values=num_col,
                                   title=f"🥧 Distribusi {num_col.title()} per {cat_col.title()}")
                        charts.append(fig)
            except Exception as e:
                st.warning(f"⚠️ Error creating pie chart: {str(e)}")
                pass
        
        # 4. Histogram untuk data numerik saja
        if numeric_cols:
            try:
                for num_col in numeric_cols[:2]:  # Maksimal 2 kolom numerik
                    fig = px.histogram(df, x=num_col, nbins=20,
                                     title=f"📊 Distribusi {num_col.title()}",
                                     labels={num_col: num_col.title()})
                    fig.update_layout(xaxis_title=num_col.title(), yaxis_title="Frequency")
                    charts.append(fig)
            except Exception as e:
                st.warning(f"⚠️ Error creating histogram: {str(e)}")
                pass
        
        return charts if charts else None
    
    except Exception as e:
        st.error(f"❌ Error membuat visualisasi: {str(e)}")
        return None

# Fungsi untuk brute force mencari query SQL
def brute_force_extract_sql(data, depth=0):
    """Fungsi rekursif untuk mencari query SQL di struktur data apapun"""
    if depth > 10:  # Prevent infinite recursion
        return None
    
    if isinstance(data, str):
        # Cek apakah string ini adalah query SQL
        if "SELECT" in data.upper() and "FROM" in data.upper() and len(data.strip()) > 15:
            return data.strip()
        
        # Coba regex patterns
        patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?)\s*```',
            r'Query:\s*(SELECT.*?)(?=\n|$)',
            r'(SELECT\s+.*?FROM\s+.*?)(?=\n\n|\n(?=[A-Z])|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, data, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if "SELECT" in match.upper() and "FROM" in match.upper():
                    return match
    
    elif isinstance(data, dict):
        # Cek semua values dalam dict
        for key, value in data.items():
            result = brute_force_extract_sql(value, depth + 1)
            if result:
                return result
    
    elif isinstance(data, (list, tuple)):
        # Cek semua items dalam list/tuple
        for item in data:
            result = brute_force_extract_sql(item, depth + 1)
            if result:
                return result
    
    elif hasattr(data, '__dict__'):
        # Cek attributes dari object
        for attr_name in dir(data):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(data, attr_name)
                    result = brute_force_extract_sql(attr_value, depth + 1)
                    if result:
                        return result
                except:
                    pass
    
    return None

# Fungsi untuk mencoba menjalankan query manual
def try_manual_query_generation(user_prompt, db):
    """Coba generate query manual berdasarkan prompt"""
    try:
        # Coba prompt yang lebih spesifik
        manual_agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=True
        )
        
        # Prompt yang lebih eksplisit
        specific_prompt = f"""
        Generate ONLY a SQL query for this request: {user_prompt}
        
        Return only the SQL query without any explanation or formatting.
        The query should be a valid PostgreSQL SELECT statement.
        """
        
        response = manual_agent.invoke({"input": specific_prompt})
        
        # Coba ekstrak dari response
        if isinstance(response, dict) and "output" in response:
            output = response["output"]
            extracted = brute_force_extract_sql(output)
            if extracted:
                return extracted
        
        return brute_force_extract_sql(response)
        
    except Exception as e:
        st.error(f"❌ Error in manual query generation: {str(e)}")
        return None

# Antarmuka Streamlit
st.title("🚀 Business Intelligence Dashboard")
st.markdown("### Analisis Data dengan AI - Gunakan Bahasa Natural untuk Query Database")

# Sidebar untuk info database
with st.sidebar:
    st.header("📊 Database Info")
    st.write(f"**Database:** {DB_NAME}")
    st.write(f"**Host:** {DB_HOST}")
    st.write(f"**Tables:** {', '.join(tables) if 'tables' in locals() else 'Loading...'}")
    
    st.header("💡 Contoh Prompt:")
    st.markdown("""
    - "Tampilkan total penjualan per kategori"
    - "Tren penjualan dalam 6 bulan terakhir"
    - "Top 10 produk terlaris"
    - "Perbandingan penjualan antar wilayah"
    - "Analisis customer berdasarkan umur"
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    user_prompt = st.text_area(
        "🔍 Masukkan pertanyaan Anda:", 
        placeholder="Contoh: Tampilkan total penjualan per kategori dari tabel sales",
        height=100
    )

with col2:
    st.markdown("### 📈 Visualisasi Otomatis")
    st.markdown("Sistem akan otomatis membuat chart berdasarkan data yang ditemukan")

if st.button("🚀 Analisis Data", type="primary"):
    if user_prompt:
        # Generate query SQL dan hasil
        try:
            # Reset callback
            sql_callback.sql_query = None
            sql_callback.all_queries = []
            sql_callback.logs = []
            sql_callback.raw_data = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Invoke agent dengan verbose logging
            status_text.text("🔄 Memproses prompt...")
            progress_bar.progress(25)
            
            response = agent.invoke({"input": user_prompt})
            progress_bar.progress(50)
            
            # Coba beberapa cara untuk mendapatkan query SQL
            sql_query = None
            
            status_text.text("🔍 Mencari query SQL...")
            
            # 1. Dari callback
            sql_query = sql_callback.get_sql_query()
            if sql_query:
                st.success(f"✅ Query ditemukan dari callback")
            
            # 2. Dari semua queries yang tertangkap
            all_queries = sql_callback.get_all_queries()
            if not sql_query and all_queries:
                sql_query = all_queries[-1]  # Ambil query terakhir
                st.success(f"✅ Query ditemukan dari all_queries")
            
            # 3. Brute force search di semua raw data
            if not sql_query:
                for raw_item in sql_callback.get_raw_data():
                    found_query = brute_force_extract_sql(raw_item)
                    if found_query:
                        sql_query = found_query
                        st.success(f"✅ Query ditemukan dari raw data")
                        break
            
            # 4. Dari intermediate steps
            if not sql_query and isinstance(response, dict) and "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    found_query = brute_force_extract_sql(step)
                    if found_query:
                        sql_query = found_query
                        st.success(f"✅ Query ditemukan dari intermediate steps")
                        break
            
            # 5. Dari output respons
            if not sql_query and isinstance(response, dict) and "output" in response:
                found_query = brute_force_extract_sql(response["output"])
                if found_query:
                    sql_query = found_query
                    st.success(f"✅ Query ditemukan dari output")
            
            # 6. Brute force search di seluruh response
            if not sql_query:
                found_query = brute_force_extract_sql(response)
                if found_query:
                    sql_query = found_query
                    st.success(f"✅ Query ditemukan dari full response")
            
            # 7. Manual query generation sebagai fallback
            if not sql_query:
                status_text.text("⚠️ Mencoba generate query manual...")
                sql_query = try_manual_query_generation(user_prompt, db)
                if sql_query:
                    st.success(f"✅ Query berhasil dibuat secara manual")
            
            progress_bar.progress(75)
            
            # Tampilkan hasil
            if sql_query:
                # Tampilkan query
                with st.expander("📋 SQL Query yang Dihasilkan", expanded=True):
                    st.code(sql_query, language="sql")
                
                # Bersihkan query dari markup
                cleaned_sql_query = re.sub(r'^```sql\s*|\s*```$', '', sql_query, flags=re.MULTILINE).strip()
                
                # Eksekusi query
                try:
                    status_text.text("⚡ Menjalankan query...")
                    
                    with engine.connect() as conn:
                        df = pd.read_sql_query(text(cleaned_sql_query), conn)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Selesai!")
                        
                        if not df.empty:
                            st.write("Data yang dihasilkan:", df)  # Debug
                            # Tampilkan hasil data
                            st.subheader("📊 Hasil Data")
                            st.dataframe(df, use_container_width=True)
                            st.success(f"✅ Query berhasil dijalankan! {len(df)} baris data ditemukan.")
                            
                            # Generate dan tampilkan visualisasi
                            st.subheader("📈 Visualisasi Data")
                            
                            with st.spinner("🎨 Membuat visualisasi..."):
                                charts = create_smart_visualization(df, user_prompt)
                                
                                if charts:
                                    # Tampilkan chart dalam tabs
                                    tab_names = [f"Chart {i+1}" for i in range(len(charts))]
                                    tabs = st.tabs(tab_names)
                                    
                                    for i, (tab, chart) in enumerate(zip(tabs, charts)):
                                        with tab:
                                            st.plotly_chart(chart, use_container_width=True)
                                    
                                    # Tampilkan summary
                                    st.info(f"📊 Berhasil membuat {len(charts)} visualisasi berdasarkan data Anda!")
                                else:
                                    st.warning("⚠️ Tidak dapat membuat visualisasi otomatis untuk data ini.")
                            
                            # Tampilkan statistik dasar
                            with st.expander("📈 Statistik Dasar Data"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Info Dataset:**")
                                    st.write(f"- Jumlah baris: {len(df)}")
                                    st.write(f"- Jumlah kolom: {len(df.columns)}")
                                    st.write(f"- Kolom: {', '.join(df.columns)}")
                                
                                with col2:
                                    st.write("**Tipe Data:**")
                                    numeric_cols, categorical_cols, date_cols = detect_column_types(df)
                                    st.write(f"- Numerik: {len(numeric_cols)}")
                                    st.write(f"- Kategorikal: {len(categorical_cols)}")
                                    st.write(f"- Tanggal: {len(date_cols)}")
                                
                                # Tampilkan describe untuk kolom numerik
                                if len(df.select_dtypes(include=['number']).columns) > 0:
                                    st.write("**Statistik Deskriptif:**")
                                    st.dataframe(df.describe())
                        else:
                            st.warning("⚠️ Query berhasil dijalankan tetapi tidak ada data yang ditemukan.")
                        
                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()
                        
                except Exception as e:
                    st.error(f"❌ Error menjalankan query: {str(e)}")
                    st.code(cleaned_sql_query, language="sql")
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.error("❌ Tidak ada query SQL yang berhasil dihasilkan.")
                progress_bar.empty()
                status_text.empty()
                
                # Tampilkan debug info lengkap
                with st.expander("🔍 Debug Information"):
                    st.write("**📋 All Queries Captured:**")
                    for i, q in enumerate(all_queries):
                        st.write(f"{i+1}. {q}")
                    
                    st.write("**📊 Response Structure:**")
                    st.json(response)
                    
                    st.write("**📝 Callback Logs:**")
                    st.text_area("Logs", sql_callback.get_logs(), height=300)
            
            # Tampilkan respons agent
            with st.expander("📝 Respons Agent"):
                if isinstance(response, dict) and "output" in response:
                    st.write(response["output"])
                else:
                    st.write(response)
                        
        except Exception as e:
            st.error(f"❌ Error dalam proses: {str(e)}")
            
            # Tampilkan debug info untuk error
            with st.expander("🔍 Error Debug Information"):
                st.write("**Error Details:**")
                st.text(str(e))
                st.write("**Callback Logs:**")
                st.text_area("Error Logs", sql_callback.get_logs(), height=300)
    else:
        st.warning("⚠️ Silakan masukkan pertanyaan terlebih dahulu.")

# Footer
st.markdown("---")
st.markdown("🚀 **Business Intelligence Dashboard** - Muhamad Wahyu Maulana")