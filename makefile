# Uso: make canvi m="tu mensaje aqui
canvi:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: falta el missatge. Usa: make canvi 'missatge a enviar'"; \
		exit 1; \
	fi
	git add .
	git commit -m "$(filter-out $@,$(MAKECMDGOALS))"

# Uso: make puja
puja:
	git push

# Extra opcional para descargar cambios: make baixa
baixa:
	git pull --rebase