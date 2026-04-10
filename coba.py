import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
import csv
from io import StringIO

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "valord_finance"
USER_ID = "user_valord"

# Model Pydantic untuk output schema
class SpendingCategory(BaseModel):
    category: str = Field(..., description="Nama kategori pengeluaran")
    amount: float = Field(..., description="Jumlah pengeluaran di kategori ini")
    percentage: Optional[float] = Field(None, description="Persentase dari total pengeluaran")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Kategori untuk rekomendasi")
    recommendation: str = Field(..., description="Detail rekomendasi")
    potential_savings: Optional[float] = Field(None, description="Estimasi penghematan bulanan")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total pengeluaran bulanan")
    monthly_income: Optional[float] = Field(None, description="Pendapatan bulanan")
    spending_categories: List[SpendingCategory] = Field(..., description="Rincian pengeluaran berdasarkan kategori")
    recommendations: List[SpendingRecommendation] = Field(..., description="Rekomendasi pengeluaran")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Jumlah dana darurat yang direkomendasikan")
    current_amount: Optional[float] = Field(None, description="Dana darurat saat ini")
    current_status: str = Field(..., description="Status penilaian dana darurat")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Kategori tabungan")
    amount: float = Field(..., description="Jumlah bulanan yang direkomendasikan")
    rationale: Optional[str] = Field(None, description="Penjelasan rekomendasi ini")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Nama teknik otomatisasi")
    description: str = Field(..., description="Cara implementasi")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Rekomendasi dana darurat")
    recommendations: List[SavingsRecommendation] = Field(..., description="Rekomendasi alokasi tabungan")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Teknik otomatisasi tabungan")

class Debt(BaseModel):
    name: str = Field(..., description="Nama hutang")
    amount: float = Field(..., description="Saldo saat ini")
    interest_rate: float = Field(..., description="Suku bunga tahunan (%)")
    min_payment: Optional[float] = Field(None, description="Pembayaran minimum bulanan")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total bunga yang dibayar")
    months_to_payoff: int = Field(..., description="Bulan hingga bebas hutang")
    monthly_payment: Optional[float] = Field(None, description="Pembayaran bulanan yang direkomendasikan")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Metode bunga tertinggi dulu")
    snowball: PayoffPlan = Field(..., description="Metode saldo terkecil dulu")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Judul rekomendasi")
    description: str = Field(..., description="Detail rekomendasi")
    impact: Optional[str] = Field(None, description="Dampak yang diharapkan")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total jumlah hutang")
    debts: List[Debt] = Field(..., description="Daftar semua hutang")
    payoff_plans: PayoffPlans = Field(..., description="Strategi pelunasan hutang")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Rekomendasi pengurangan hutang")

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def parse_json_safely(data: str, default_value: Any = None) -> Any:
    """Parsing JSON dengan penanganan error yang aman"""
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value

class ValordFinanceSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()
        
        self.budget_analysis_agent = LlmAgent(
            name="BudgetAnalysisAgent",
            model="gemini-2.5-flash",
            description="Menganalisis data keuangan untuk mengkategorikan pola pengeluaran dan merekomendasikan perbaikan anggaran",
            instruction="""Anda adalah Agen Analisis Anggaran yang mengkhususkan diri dalam meninjau transaksi dan pengeluaran keuangan.
Anda adalah agen pertama dalam rangkaian tiga agen penasihat keuangan.

Tugas Anda:
1. Analisis pendapatan, transaksi, dan pengeluaran secara detail
2. Kategorikan pengeluaran ke dalam grup logis dengan rincian yang jelas
3. Identifikasi pola dan tren pengeluaran di berbagai kategori
4. Sarankan area spesifik dimana pengeluaran bisa dikurangi dengan saran konkret
5. Berikan rekomendasi yang dapat ditindaklanjuti dengan jumlah penghematan yang terukur

Pertimbangkan:
- Jumlah tanggungan saat mengevaluasi pengeluaran rumah tangga
- Rasio pengeluaran tipikal untuk tingkat pendapatan (perumahan 30%, makanan 15%, dll)
- Pengeluaran esensial vs diskresioner dengan pemisahan yang jelas
- Pola pengeluaran musiman jika data mencakup beberapa bulan

Untuk kategori pengeluaran, sertakan SEMUA pengeluaran dari data pengguna, pastikan persentase total 100%,
dan pastikan setiap pengeluaran terkategorikan.

Untuk rekomendasi:
- Berikan minimal 3-5 rekomendasi spesifik dan dapat ditindaklanjuti dengan estimasi penghematan
- Jelaskan alasan di balik setiap rekomendasi
- Pertimbangkan dampak pada kualitas hidup dan kesehatan keuangan jangka panjang
- Sarankan langkah implementasi spesifik untuk setiap rekomendasi

PENTING: Simpan analisis Anda di state['budget_analysis'] untuk digunakan agen berikutnya.""",
            output_schema=BudgetAnalysis,
            output_key="budget_analysis"
        )
        
        self.savings_strategy_agent = LlmAgent(
            name="SavingsStrategyAgent",
            model="gemini-2.5-flash",
            description="Merekomendasikan strategi tabungan optimal berdasarkan pendapatan, pengeluaran, dan tujuan keuangan",
            instruction="""Anda adalah Agen Strategi Tabungan yang mengkhususkan diri dalam membuat rencana tabungan personal.
Anda adalah agen kedua dalam rangkaian. BACA analisis anggaran dari state['budget_analysis'] terlebih dahulu.

Tugas Anda:
1. Tinjau hasil analisis anggaran dari state['budget_analysis']
2. Rekomendasikan strategi tabungan komprehensif berdasarkan analisis
3. Hitung ukuran dana darurat optimal berdasarkan pengeluaran dan tanggungan
4. Sarankan alokasi tabungan yang tepat untuk berbagai tujuan
5. Rekomendasikan teknik otomatisasi praktis untuk menabung secara konsisten

Pertimbangkan:
- Faktor risiko berdasarkan stabilitas pekerjaan dan tanggungan
- Keseimbangan kebutuhan segera dengan kesehatan keuangan jangka panjang
- Tingkat tabungan progresif seiring bertambahnya pendapatan diskresioner
- Berbagai tujuan tabungan (dana darurat, pensiun, pembelian spesifik)
- Area potensi penghematan yang diidentifikasi dalam analisis anggaran

PENTING: Simpan strategi Anda di state['savings_strategy'] untuk digunakan Agen Pengurangan Hutang.""",
            output_schema=SavingsStrategy,
            output_key="savings_strategy"
        )
        
        self.debt_reduction_agent = LlmAgent(
            name="DebtReductionAgent",
            model="gemini-2.5-flash",
            description="Membuat rencana pelunasan hutang optimal untuk meminimalkan bunga dan waktu bebas hutang",
            instruction="""Anda adalah Agen Pengurangan Hutang yang mengkhususkan diri dalam membuat strategi pelunasan hutang.
Anda adalah agen terakhir dalam rangkaian. BACA state['budget_analysis'] dan state['savings_strategy'] terlebih dahulu.

Tugas Anda:
1. Tinjau analisis anggaran dan strategi tabungan dari state
2. Analisis hutang berdasarkan suku bunga, saldo, dan pembayaran minimum
3. Buat rencana pelunasan hutang prioritas (metode avalanche dan snowball)
4. Hitung total bunga yang dibayar dan waktu bebas hutang
5. Sarankan peluang konsolidasi atau refinancing hutang
6. Berikan rekomendasi spesifik untuk mempercepat pelunasan hutang

Pertimbangkan:
- Kendala arus kas dari analisis anggaran
- Tujuan dana darurat dan tabungan dari strategi tabungan
- Faktor psikologis (kemenangan cepat vs optimasi matematis)
- Dampak skor kredit dan peluang perbaikan

PENTING: Simpan rencana final Anda di state['debt_reduction'] dan pastikan selaras dengan analisis sebelumnya.""",
            output_schema=DebtReduction,
            output_key="debt_reduction"
        )
        
        self.coordinator_agent = SequentialAgent(
            name="ValordFinanceCoordinator",
            description="Mengkoordinasikan agen keuangan khusus untuk memberikan saran keuangan komprehensif",
            sub_agents=[
                self.budget_analysis_agent,
                self.savings_strategy_agent,
                self.debt_reduction_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = f"valord_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            initial_state = {
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants": financial_data.get("dependants", 0),
                "transactions": financial_data.get("transactions", []),
                "manual_expenses": financial_data.get("manual_expenses", {}),
                "debts": financial_data.get("debts", [])
            }
            
            session = self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=initial_state
            )
            
            if session.state.get("transactions"):
                self._preprocess_transactions(session)
            
            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)
            
            default_results = self._create_default_results(financial_data)
            
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(financial_data))]
            )
            
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break
            
            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during finance analysis: {str(e)}")
            raise
        finally:
            self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
    
    def _preprocess_transactions(self, session):
        transactions = session.state.get("transactions", [])
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        if 'Category' in df.columns and 'Amount' in df.columns:
            category_spending = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["category_spending"] = category_spending
            session.state["total_spending"] = df['Amount'].sum()
    
    def _preprocess_manual_expenses(self, session):
        manual_expenses = session.state.get("manual_expenses", {})
        if not manual_expenses or manual_expenses is None:
            return
        
        session.state.update({
            "total_manual_spending": sum(manual_expenses.values()),
            "manual_category_spending": manual_expenses
        })

    def _create_default_results(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        monthly_income = financial_data.get("monthly_income", 0)
        expenses = financial_data.get("manual_expenses", {})
        
        if expenses is None:
            expenses = {}
        
        if not expenses and financial_data.get("transactions"):
            expenses = {}
            for transaction in financial_data["transactions"]:
                category = transaction.get("Category", "Belum Dikategorikan")
                amount = transaction.get("Amount", 0)
                expenses[category] = expenses.get(category, 0) + amount
        
        total_expenses = sum(expenses.values())
        
        return {
            "budget_analysis": {
                "total_expenses": total_expenses,
                "monthly_income": monthly_income,
                "spending_categories": [
                    {"category": cat, "amount": amt, "percentage": (amt / total_expenses * 100) if total_expenses > 0 else 0}
                    for cat, amt in expenses.items()
                ],
                "recommendations": [
                    {"category": "Umum", "recommendation": "Pertimbangkan untuk meninjau pengeluaran Anda dengan teliti", "potential_savings": total_expenses * 0.1}
                ]
            },
            "savings_strategy": {
                "emergency_fund": {
                    "recommended_amount": total_expenses * 6,
                    "current_amount": 0,
                    "current_status": "Belum Dimulai"
                },
                "recommendations": [
                    {"category": "Dana Darurat", "amount": total_expenses * 0.1, "rationale": "Bangun dana darurat terlebih dahulu"},
                    {"category": "Pensiun", "amount": monthly_income * 0.15, "rationale": "Tabungan jangka panjang"}
                ],
                "automation_techniques": [
                    {"name": "Transfer Otomatis", "description": "Atur transfer otomatis saat gajian"}
                ]
            },
            "debt_reduction": {
                "total_debt": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])),
                "debts": financial_data.get("debts", []),
                "payoff_plans": {
                    "avalanche": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.2,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24
                    },
                    "snowball": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.25,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24
                    }
                },
                "recommendations": [
                    {"title": "Tingkatkan Pembayaran", "description": "Tingkatkan pembayaran bulanan Anda", "impact": "Mengurangi total bunga yang dibayar"}
                ]
            }
        }

def display_budget_analysis(analysis: Dict[str, Any]):
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("❌ Gagal memparsing hasil analisis anggaran")
            return
    
    if not isinstance(analysis, dict):
        st.error("❌ Format analisis anggaran tidak valid")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Pengeluaran Berdasarkan Kategori")
        if "spending_categories" in analysis:
            fig = px.pie(
                values=[cat["amount"] for cat in analysis["spending_categories"]],
                names=[cat["category"] for cat in analysis["spending_categories"]],
                title="Rincian Pengeluaran Anda",
                color_discrete_sequence=px.colors.sequential.Sunset
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "total_expenses" in analysis:
            income = analysis.get("monthly_income", 0)
            expenses = analysis["total_expenses"]
            surplus_deficit = income - expenses
            
            st.subheader("💰 Pendapatan vs Pengeluaran")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Pendapatan", "Pengeluaran"], 
                                y=[income, expenses],
                                marker_color=["#10B981", "#EF4444"],
                                text=[f"Rp{income:,.0f}", f"Rp{expenses:,.0f}"], 
                                textposition='auto'))
            fig.update_layout(title="Perbandingan Bulanan", showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("💵 Surplus/Defisit Bulanan", f"Rp{surplus_deficit:,.0f}", 
            delta=f"Rp{surplus_deficit:,.0f}",
            delta_color="normal")

    if "recommendations" in analysis:
        st.subheader("🎯 Rekomendasi Pengurangan Pengeluaran")
        for i, rec in enumerate(analysis["recommendations"], 1):
            with st.container():
                st.markdown(f"**{i}. {rec['category']}**")
                st.markdown(f"*{rec['recommendation']}*")
                if "potential_savings" in rec:
                    st.success(f"💰 Penghematan Potensial: **Rp{rec['potential_savings']:,.0f}/bulan**")

def display_savings_strategy(strategy: Dict[str, Any]):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("❌ Gagal memparsing hasil strategi tabungan")
            return
    
    if not isinstance(strategy, dict):
        st.error("❌ Format strategi tabungan tidak valid")
        return
    
    st.subheader("💎 Strategi Tabungan Valord")
    
    if "emergency_fund" in strategy:
        ef = strategy["emergency_fund"]
        st.markdown("### 🛡️ Dana Darurat")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Ukuran Direkomendasikan", f"Rp{ef['recommended_amount']:,.0f}")
        with col2:
            st.metric("Status Saat Ini", ef['current_status'])
        
        if "current_amount" in ef and "recommended_amount" in ef:
            progress = min(ef["current_amount"] / ef["recommended_amount"], 1.0)
            st.progress(progress)
            st.caption(f"Rp{ef['current_amount']:,.0f} / Rp{ef['recommended_amount']:,.0f}")
    
    if "recommendations" in strategy:
        st.markdown("### 📈 Alokasi Tabungan Bulanan")
        for rec in strategy["recommendations"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{rec['category']}**")
                st.caption(rec['rationale'])
            with col2:
                st.metric("Jumlah", f"Rp{rec['amount']:,.0f}")
    
    if "automation_techniques" in strategy:
        st.markdown("### 🤖 Teknik Otomatisasi Tabungan")
        for technique in strategy["automation_techniques"]:
            st.info(f"**{technique['name']}**: {technique['description']}")

def display_debt_reduction(plan: Dict[str, Any]):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("❌ Gagal memparsing hasil pengurangan hutang")
            return
    
    if not isinstance(plan, dict):
        st.error("❌ Format pengurangan hutang tidak valid")
        return
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if "total_debt" in plan:
            st.metric("💳 Total Hutang", f"Rp{plan['total_debt']:,.0f}")
    
    if "debts" in plan and plan["debts"]:
        st.subheader("📋 Daftar Hutang Anda")
        debt_df = pd.DataFrame(plan["debts"])
        st.dataframe(debt_df, use_container_width=True)
        
        fig = px.bar(debt_df, x="name", y="amount", color="interest_rate",
                    labels={"name": "Hutang", "amount": "Jumlah (Rp)", "interest_rate": "Suku Bunga (%)"},
                    title="Rincian Hutang", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    
    if "payoff_plans" in plan:
        st.subheader("⚡ Rencana Pelunasan Hutang")
        tabs = st.tabs(["🔥 Metode Avalanche", "❄️ Metode Snowball", "⚖️ Perbandingan"])
        
        with tabs[0]:
            st.markdown("### Avalanche (Bunga Tertinggi Dulu)")
            if "avalanche" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Bunga", f"Rp{avalanche['total_interest']:,.0f}")
                with col2:
                    st.metric("Waktu Pelunasan", f"{avalanche['months_to_payoff']} bulan")
                if "monthly_payment" in avalanche:
                    st.info(f"💳 **Pembayaran Bulanan**: Rp{avalanche['monthly_payment']:,.0f}")
        
        with tabs[1]:
            st.markdown("### Snowball (Saldo Terkecil Dulu)")
            if "snowball" in plan["payoff_plans"]:
                snowball = plan["payoff_plans"]["snowball"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Bunga", f"Rp{snowball['total_interest']:,.0f}")
                with col2:
                    st.metric("Waktu Pelunasan", f"{snowball['months_to_payoff']} bulan")
                if "monthly_payment" in snowball:
                    st.info(f"💳 **Pembayaran Bulanan**: Rp{snowball['monthly_payment']:,.0f}")
        
        with tabs[2]:
            st.markdown("### Perbandingan Metode")
            if "avalanche" in plan["payoff_plans"] and "snowball" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                snowball = plan["payoff_plans"]["snowball"]
                
                comparison_data = {
                    "Metode": ["Avalanche", "Snowball"],
                    "Total Bunga": [avalanche["total_interest"], snowball["total_interest"]],
                    "Bulan Pelunasan": [avalanche["months_to_payoff"], snowball["months_to_payoff"]]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                fig = go.Figure(data=[
                    go.Bar(name="Total Bunga", x=comparison_df["Metode"], y=comparison_df["Total Bunga"], marker_color="#EF4444"),
                    go.Bar(name="Bulan", x=comparison_df["Metode"], y=comparison_df["Bulan Pelunasan"], marker_color="#3B82F6")
                ])
                fig.update_layout(barmode='group', title="Perbandingan Strategi Pelunasan", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    if "recommendations" in plan:
        st.subheader("🚀 Rekomendasi Pengurangan Hutang")
        for rec in plan["recommendations"]:
            st.success(f"**{rec['title']}**")
            st.markdown(f"*{rec['description']}*")
            if "impact" in rec:
                st.caption(f"📈 Dampak: {rec['impact']}")

def parse_csv_transactions(file_content) -> List[Dict[str, Any]]:
    """Parse file CSV menjadi daftar transaksi"""
    try:
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Kolom wajib hilang: {', '.join(missing_columns)}")
        
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        
        category_totals = df.groupby('Category')['Amount'].sum().reset_index()
        transactions = df.to_dict('records')
        
        return {
            'transactions': transactions,
            'category_totals': category_totals.to_dict('records')
        }
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

def validate_csv_format(file) -> bool:
    """Validasi format file CSV"""
    try:
        content = file.read().decode('utf-8')
        dialect = csv.Sniffer().sniff(content)
        has_header = csv.Sniffer().has_header(content)
        file.seek(0)
        
        if not has_header:
            return False, "File CSV harus memiliki header"
            
        df = pd.read_csv(StringIO(content))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Kolom wajib hilang: {', '.join(missing_columns)}"
            
        try:
            pd.to_datetime(df['Date'])
        except:
            return False, "Format tanggal tidak valid di kolom Date"
            
        try:
            df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        except:
            return False, "Format jumlah tidak valid di kolom Amount"
            
        return True, "Format CSV valid"
    except Exception as e:
        return False, f"Format CSV tidak valid: {str(e)}"

def display_csv_preview(df: pd.DataFrame):
    """Tampilkan preview data CSV dengan statistik dasar"""
    st.subheader("👀 Preview Data CSV")
    
    total_transactions = len(df)
    total_amount = df['Amount'].sum()
    
    df_dates = pd.to_datetime(df['Date'])
    date_range = f"{df_dates.min().strftime('%d/%m/%Y')} - {df_dates.max().strftime('%d/%m/%Y')}"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Transaksi", total_transactions)
    with col2:
        st.metric("💰 Total Jumlah", f"Rp{total_amount:,.0f}")
    with col3:
        st.metric("📅 Rentang Tanggal", date_range)
    
    st.subheader("📈 Pengeluaran per Kategori")
    category_totals = df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['Kategori', 'Total (Rp)', 'Jumlah Transaksi']
    category_totals['Total (Rp)'] = category_totals['Total (Rp)'].round(0)
    st.dataframe(category_totals, use_container_width=True)
    
    st.subheader("📄 Sampel Transaksi")
    st.dataframe(df.head(10), use_container_width=True)

def main():
    # Konfigurasi halaman dengan tema Valord Finance
    st.set_page_config(
        page_title="Valord Finance - AI Penasihat Keuangan",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS custom untuk tema Valord Finance
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
    }
    .valord-card {
        background: linear-gradient(145deg, #f0f2f5, #ffffff);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header dengan branding Valord Finance
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">💎 VALORD FINANCE</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #64748b;">AI Penasihat Keuangan Pribadi Anda</h3>', unsafe_allow_html=True)
    
    # Sidebar dengan branding lengkap
    with st.sidebar:
        st.markdown("## 🏦 Valord Finance")
        st.markdown("---")
        st.image("Valord.png", width=250, use_column_width=True)
        st.markdown("### ✨ Fitur Unggulan")
        st.markdown("""
        - 📊 Analisis Anggaran Otomatis
        - 💰 Strategi Tabungan Personal
        - ⚡ Rencana Pelunasan Hutang
        - 🤖 Agen AI Khusus
        """)
        st.markdown("---")
        
        st.subheader("📋 Template CSV")
        st.markdown("""
        **Format yang dibutuhkan:**
        - Tanggal (DD/MM/YYYY)
        - Kategori
        - Jumlah (angka)
        """)
        
        sample_csv = """Tanggal,Kategori,Jumlah
01/01/2024,Rumah,1200000
02/01/2024,Makanan,150500
03/01/2024,Transportasi,45000"""
        
        st.download_button(
            label="⬇️ Download Template",
            data=sample_csv,
            file_name="template_valord_finance.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Cek API Key
    if not GEMINI_API_KEY:
        st.error("🔑 **GOOGLE_API_KEY** tidak ditemukan di file .env")
        st.info("Tambahkan `GOOGLE_API_KEY=your_key` ke file `.env` Anda")
        return
    
    st.markdown("---")
    
    # Tabs utama
    tab1, tab2, tab3 = st.tabs(["💼 Data Keuangan", "📈 Hasil Analisis", "ℹ️ Tentang Valord"])
    
    with tab1:
        # Input Pendapatan & Tanggungan
        with st.container():
            st.markdown('<div class="valord-card">', unsafe_allow_html=True)
            st.subheader("💰 Pendapatan & Rumah Tangga")
            col1, col2 = st.columns([2, 1])
            with col1:
                monthly_income = st.number_input(
                    "Pendapatan Bulanan (Rp)",
                    min_value=0.0,
                    step=50000.0,
                    value=5000000.0,
                    format="%.0f",
                    help="Masukkan total pendapatan bersih bulanan"
                )
            with col2:
                dependants = st.number_input(
                    "Jumlah Tanggungan",
                    min_value=0,
                    step=1,
                    value=0,
                    help="Termasuk anak, orang tua, dll"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input Pengeluaran
        with st.container():
            st.markdown('<div class="valord-card">', unsafe_allow_html=True)
            st.subheader("💳 Pengeluaran Bulanan")
            expense_option = st.radio(
                "Pilih cara input pengeluaran:",
                ("📤 Upload CSV", "✍️ Manual"),
                horizontal=True
            )
            
            transaction_file = None
            manual_expenses = {}
            transactions_df = None
            
            if expense_option == "📤 Upload CSV":
                col1, col2 = st.columns([3, 1])
                with col1:
                    transaction_file = st.file_uploader(
                        "Upload file CSV transaksi",
                        type=["csv"],
                        help="File CSV dengan kolom: Tanggal, Kategori, Jumlah"
                    )
                
                if transaction_file:
                    is_valid, message = validate_csv_format(transaction_file)
                    if is_valid:
                        transaction_file.seek(0)
                        file_content = transaction_file.read()
                        parsed_data = parse_csv_transactions(file_content)
                        transactions_df = pd.DataFrame(parsed_data['transactions'])
                        display_csv_preview(transactions_df)
                        st.success("✅ File berhasil diupload!")
                    else:
                        st.error(message)
            else:
                st.markdown("### Kategori Pengeluaran Standar")
                categories = [
                    ("🏠 Rumah", "Rumah"),
                    ("🔌 Utilitas", "Utilitas"),
                    ("🍽️ Makanan", "Makanan"),
                    ("🚗 Transportasi", "Transportasi"),
                    ("🏥 Kesehatan", "Kesehatan"),
                    ("🎭 Hiburan", "Hiburan"),
                    ("👤 Pribadi", "Pribadi"),
                    ("💰 Tabungan", "Tabungan"),
                    ("📦 Lainnya", "Lainnya")
                ]
                
                cols = st.columns(3)
                for i, (emoji_cat, cat) in enumerate(categories):
                    with cols[i % 3]:
                        manual_expenses[cat] = st.number_input(
                            emoji_cat,
                            min_value=0.0,
                            step=25000.0,
                            value=0.0,
                            format="%.0f",
                            key=f"manual_{cat}"
                        )
                
                if any(manual_expenses.values()):
                    total_manual = sum(v for v in manual_expenses.values() if v > 0)
                    st.metric("📊 Total Pengeluaran Manual", f"Rp{total_manual:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input Hutang
        with st.container():
            st.markdown('<div class="valord-card">', unsafe_allow_html=True)
            st.subheader("🏦 Informasi Hutang")
            num_debts = st.number_input(
                "Jumlah hutang:",
                min_value=0,
                max_value=10,
                value=0
            )
            
            debts = []
            if num_debts > 0:
                for i in range(num_debts):
                    st.markdown(f"### Hutang #{i+1}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        debt_name = st.text_input(f"Nama Hutang {i+1}", value=f"Hutang {i+1}", key=f"dname_{i}")
                    with col2:
                        debt_amount = st.number_input(f"Jumlah (Rp)", min_value=0.0, value=1000000.0, format="%.0f", key=f"damount_{i}")
                    with col3:
                        interest_rate = st.number_input(f"Bunga (%)", min_value=0.0, value=10.0, step=0.5, key=f"drate_{i}")
                    
                    debts.append({
                        "name": debt_name,
                        "amount": debt_amount,
                        "interest_rate": interest_rate,
                        "min_payment": debt_amount * 0.05  # Default 5%
                    })
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tombol Analisis
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 ANALISIS KEUANGAN VALORD", use_container_width=True, type="primary"):
                financial_data = {
                    "monthly_income": monthly_income,
                    "dependants": dependants,
                    "transactions": transactions_df.to_dict('records') if 'transactions_df' in locals() and transactions_df is not None else [],
                    "manual_expenses": manual_expenses if expense_option == "✍️ Manual" else {},
                    "debts": debts
                }
                
                with st.spinner("🤖 Valord AI sedang menganalisis keuangan Anda..."):
                    finance_system = ValordFinanceSystem()
                    results = asyncio.run(finance_system.analyze_finances(financial_data))
                
                st.session_state.results = results
                st.rerun()
    
    with tab2:
        if 'results' in st.session_state and st.session_state.results:
            results = st.session_state.results
            analysis_tabs = st.tabs(["💰 Analisis Anggaran", "💎 Strategi Tabungan", "⚡ Pengurangan Hutang"])
            
            with analysis_tabs[0]:
                display_budget_analysis(results.get("budget_analysis", {}))
            
            with analysis_tabs[1]:
                display_savings_strategy(results.get("savings_strategy", {}))
            
            with analysis_tabs[2]:
                display_debt_reduction(results.get("debt_reduction", {}))
        else:
            st.info("📊 Masukkan data keuangan Anda di tab pertama untuk melihat hasil analisis")
    
    with tab3:
        st.markdown("""
        ### Tentang AI Financial Coach

        Aplikasi ini menggunakan **Google's Agent Development Kit (ADK)** untuk memberikan analisis dan saran keuangan secara komprehensif melalui beberapa agen AI yang memiliki peran khusus:

        1. **🔍 Agen Analisis Anggaran**
        - Menganalisis pola pengeluaran
        - Mengidentifikasi area yang bisa menghemat biaya
        - Memberikan rekomendasi yang dapat langsung diterapkan

        2. **💰 Agen Strategi Tabungan**
        - Membuat rencana tabungan yang dipersonalisasi
        - Menghitung kebutuhan dana darurat
        - Memberikan saran otomatisasi tabungan

        3. **💳 Agen Pengelolaan Hutang**
        - Menyusun strategi pelunasan hutang yang optimal
        - Membandingkan berbagai metode pembayaran hutang
        - Memberikan tips praktis untuk mengurangi hutang

        ### Privasi & Keamanan

        - Semua data diproses secara lokal
        - Tidak ada informasi keuangan yang disimpan atau dikirim
        - Komunikasi API dengan layanan Google dilakukan secara aman

        ### Butuh Bantuan?

        Untuk dukungan atau pertanyaan:
        - Lihat dokumentasi di:  
        https://github.com/Shubhamsaboo/awesome-llm-apps

        - Laporkan masalah melalui GitHub:  
        https://github.com/Shubhamsaboo/awesome-llm-apps/issues
        """)
if __name__ == "__main__":
    main()
    
        